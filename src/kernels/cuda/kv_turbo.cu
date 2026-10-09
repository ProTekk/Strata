// src/kernels/cuda/kv_turbo.cu - TurboQuant KV storage (see include/strata/kernels/kv_turbo.hpp).
//
// Ported from buun's TurboQuant (ggml/src/ggml-cuda/turbo-quant-cuda.cuh, turbo-wht.cu, turbo-tcq-alpha.cuh).
// Dropped in the port: the PFHEAD/InnerQ/meansub/extract probes and the runtime codebook book-swaps (the tables
// are compile-time constants here - one table set, no cudaMemcpyToSymbol, no desync class). Kept bit-exact, on
// pain of KV chaos: the seed-42 sign arrays, the codebooks, the encode pipeline order, the Viterbi tie-breaks
// (strict < keeps the LOWEST predecessor / lowest final state - exactly buun's warp-argmin order), and the norm
// reduction trees (kv_turbo_parity replays them on the host).
//
// Layout: head_dim 256 = two 128-value rotation groups; a cell's K (or V) for one head is two blocks back to
// back. Inline fp16 norm, no scale arrays - one contiguous run per pool per page.
//
// One kernel family serves both writers: with `step != nullptr` it is the decode/verify append (positions from
// the step buffer, planes = K|V), with `step == nullptr` the prompt append (pos = pos0 + t over the [T, heads,
// dim] arrays, plus the optional staging pool). The addressing is the same expression either way.
#include "strata/kernels/kv_turbo.hpp"
#include "strata/kernels/kv_turbo_tables.hpp"
#include "strata/kernels/kv_turbo_alpha.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

int g_kv_turbo_tier = kTurbo4;  // the engine's active tier (generate.cpp sets it from --kv)

namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_turbo: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The single device-global table set. buun duplicated these as per-TU __constant__ copies and swapped some at
// runtime (two documented desync sources, turbo-tcq-alpha.cuh:7); here they are constexpr in one header,
// uploaded once per process.
__constant__ float c_turbo_s1[128] = {0};
__constant__ float c_turbo_s2[128] = {0};
__constant__ float c_turbo4_centroids[16] = {0};
__constant__ float c_turbo4_mids[15] = {0};
__constant__ float c_turbo8_centroids[256] = {0};
__constant__ float c_turbo3_tcq_cb[512] = {0};
__constant__ float c_turbo3_tcq_cb_v[512] = {0};
__constant__ float c_turbo2_tcq_cb[256] = {0};
__constant__ float c_turbo2_tcq_cb_v[256] = {0};
__constant__ float c_turbo1_tcq_cb[256] = {0};
__constant__ float c_turbo1_tcq_cb_v[256] = {0};

void upload_tables() {
    static bool done = false;
    if (done) return;
    cudaMemcpyToSymbol(c_turbo_s1, kTurboWhtS1, sizeof(kTurboWhtS1));
    cudaMemcpyToSymbol(c_turbo_s2, kTurboWhtS2, sizeof(kTurboWhtS2));
    cudaMemcpyToSymbol(c_turbo4_centroids, kTurbo4Centroids, sizeof(kTurbo4Centroids));
    cudaMemcpyToSymbol(c_turbo4_mids, kTurbo4Mids, sizeof(kTurbo4Mids));
    cudaMemcpyToSymbol(c_turbo8_centroids, kTurbo8Centroids, sizeof(kTurbo8Centroids));
    cudaMemcpyToSymbol(c_turbo3_tcq_cb, kTurbo3TcqCodebook, sizeof(kTurbo3TcqCodebook));
    cudaMemcpyToSymbol(c_turbo3_tcq_cb_v, kTurbo3TcqCodebookV, sizeof(kTurbo3TcqCodebookV));
    cudaMemcpyToSymbol(c_turbo2_tcq_cb, kTurbo2TcqCodebook, sizeof(kTurbo2TcqCodebook));
    cudaMemcpyToSymbol(c_turbo2_tcq_cb_v, kTurbo2TcqCodebookV, sizeof(kTurbo2TcqCodebookV));
    cudaMemcpyToSymbol(c_turbo1_tcq_cb, kTurbo1TcqCodebook, sizeof(kTurbo1TcqCodebook));
    cudaMemcpyToSymbol(c_turbo1_tcq_cb_v, kTurbo1TcqCodebookV, sizeof(kTurbo1TcqCodebookV));
    check("table upload");
    done = true;
}

constexpr float kInvSqrt128 = 0.08838834764831845f;

// ---- device helpers ----

// Sequential orthonormal FWHT-128 over 128 registers (buun's turbo_fwht_128_cuda verbatim). The butterfly
// stage order fixes each output's operation tree; the warp-shuffle form below runs the SAME stage sequence,
// so the two are bitwise equal.
__device__ __forceinline__ void turbo_fwht128_regs(float* x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 128; i++) x[i] *= kInvSqrt128;
}

// Forward rotation x*s1 -> FWHT-128 -> *s2 (turbo-quant-cuda.cuh:779-783).
__device__ __forceinline__ void turbo_rotate_forward_regs(float* x) {
#pragma unroll
    for (int i = 0; i < 128; i++) x[i] *= c_turbo_s1[i];
    turbo_fwht128_regs(x);
#pragma unroll
    for (int i = 0; i < 128; i++) x[i] *= c_turbo_s2[i];
}

// Nearest of the 16 Lloyd-Max centroids: binary search over the 15 midpoints (turbo_find_nearest_4bit verbatim).
__device__ __forceinline__ uint8_t turbo_find_nearest_4bit(float val) {
    if (val < c_turbo4_mids[7]) {
        if (val < c_turbo4_mids[3]) {
            if (val < c_turbo4_mids[1]) return val < c_turbo4_mids[0] ? 0 : 1;
            return val < c_turbo4_mids[2] ? 2 : 3;
        }
        if (val < c_turbo4_mids[5]) return val < c_turbo4_mids[4] ? 4 : 5;
        return val < c_turbo4_mids[6] ? 6 : 7;
    }
    if (val < c_turbo4_mids[11]) {
        if (val < c_turbo4_mids[9]) return val < c_turbo4_mids[8] ? 8 : 9;
        return val < c_turbo4_mids[10] ? 10 : 11;
    }
    if (val < c_turbo4_mids[13]) return val < c_turbo4_mids[12] ? 12 : 13;
    return val < c_turbo4_mids[14] ? 14 : 15;
}

// TCQ bitstream read: the L-bit state of element t lives at bit t*k, LSB-first (buun's read_9_bits for
// 3_tcq, read_8_bits for 2_tcq / 1_tcq). The stored (L-k)-bit init prefix cancels against the state
// window: state_t spans outputs t-(L/k-1) .. t, whose first bit sits at (L-k) + (t-(L/k-1))*k = t*k.
__device__ __forceinline__ int turbo_tcq_read_state(const uint8_t* qs, int t, int k) {
    const int bit_pos = t * k;
    const int byte_idx = bit_pos >> 3;
    const int bit_off = bit_pos & 7;
    // 9 bits at bit_off up to 7 can span 3 bytes; 8 bits span at most 2.
    uint32_t raw = (uint32_t) qs[byte_idx] | ((uint32_t) qs[byte_idx + 1] << 8);
    if (k == 3) raw |= (uint32_t) qs[byte_idx + 2] << 16;
    return (int) ((raw >> bit_off) & ((1u << (k == 3 ? 9 : 8)) - 1u));
}

__device__ __forceinline__ const float* turbo_tcq_cb(int tier, int is_v) {
    if (tier == kTurbo3Tcq) return is_v ? c_turbo3_tcq_cb_v : c_turbo3_tcq_cb;
    if (tier == kTurbo2Tcq) return is_v ? c_turbo2_tcq_cb_v : c_turbo2_tcq_cb;
    return is_v ? c_turbo1_tcq_cb_v : c_turbo1_tcq_cb;
}

__device__ __forceinline__ float turbo_tcq_decode_alpha(int tier, int is_v) {
    if (!is_v) return 1.0f;
    return tier == kTurbo3Tcq ? kTurboTcqAlphaVT3 : (tier == kTurbo2Tcq ? kTurboTcqAlphaVT2 : kTurboTcqAlphaVT1);
}

// ---- rotation kernels (verbatim port of buun's k_turbo_wht) ----

__global__ void turbo_rotate128_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       const int64_t n_elements, const int direction) {
    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const float* s_first  = (direction == 0) ? c_turbo_s1 : c_turbo_s2;
    const float* s_second = (direction == 0) ? c_turbo_s2 : c_turbo_s1;

    __shared__ float buf[128];

    if (threadIdx.x < 128) buf[threadIdx.x] = src[offset + threadIdx.x] * s_first[threadIdx.x];
    __syncthreads();

    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    if (threadIdx.x < 128) dst[offset + threadIdx.x] = buf[threadIdx.x] * kInvSqrt128 * s_second[threadIdx.x];
}

// ---- vanilla encode (turbo8_0 / turbo4_0): the whole 128-group in one thread's registers ----
// buun runs this one-BLOCK-per-group with the group in shared memory; the math is per-thread sequential
// (norm, rotate, quantize, norm-correct), so thread-per-group is bitwise identical. Thread 0 of the block
// does the work (the block shape matches the TCQ kernel's grid; the other threads only exist for it).

template <int TIER>
__device__ __forceinline__ void turbo_quantize_vanilla_group(const float* __restrict__ src, uint8_t* __restrict__ dst) {
    float x[128];
    float norm_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
#pragma unroll
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward_regs(x);

    if (TIER == kTurbo4) {
        auto* blk = reinterpret_cast<block_turbo4_0*>(dst);
#pragma unroll
        for (int j = 0; j < 128; j += 2) {
            const uint8_t idx0 = turbo_find_nearest_4bit(x[j]);
            const uint8_t idx1 = turbo_find_nearest_4bit(x[j + 1]);
            blk->qs[j / 2] = (uint8_t) ((idx1 << 4) | idx0);
        }
        // Norm correction in the rotated space (sequential read-back of the packed codes, buun's order).
        float recon_sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 128; j++) {
            const uint8_t idx = (j & 1) ? (blk->qs[j / 2] >> 4) : (blk->qs[j / 2] & 0xF);
            const float r = c_turbo4_centroids[idx];
            recon_sq += r * r;
        }
        const float recon_norm = sqrtf(recon_sq);
        blk->norm = f16_from_f32((recon_norm > 1e-10f) ? norm / recon_norm : norm);
    } else {
        // turbo8_0: per-block absmax scale + uniform 256-grid (rotated-Q8; the Hadamard makes the tails tame).
        auto* blk = reinterpret_cast<block_turbo8_0*>(dst);
        float absmax = 0.0f;
#pragma unroll
        for (int j = 0; j < 128; j++) absmax = fmaxf(absmax, fabsf(x[j]));
        const float scale = absmax > 1e-10f ? absmax : 1e-10f;
        const float inv_scale = 1.0f / scale;
#pragma unroll
        for (int j = 0; j < 128; j++) {
            int idx = (int) lrintf(x[j] * inv_scale * 127.5f + 127.5f);
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            blk->qs[j] = (uint8_t) idx;
        }
        blk->norm = f16_from_f32(norm * scale);
    }
}

// ---- TCQ encode: one block per 128-group, 256 threads (buun's k_set_rows_turbo{2,3}_tcq shape) ----
// Right-shift bitshift trellis: ns = (prev >> k) | (out << (L-k)). The best predecessor of a state depends
// only on the state's low (L-k) bits, so 1<<(L-k) predecessor minima cover all states. Tie-break: strict <
// keeps the LOWEST predecessor index; the final argmin keeps the lowest state index - buun's warp-argmin
// order, so paths match bitwise.

struct TcqParams {
    int k;            // output bits per step
    int L;            // state width in bits
    int states;       // 1 << L
    int bt_groups;    // 1 << (L - k): predecessor-minimum groups
    int prefix_bits;  // initial-state bits stored before the outputs (== L - k)
    int qs_bytes;
};
__host__ __device__ __forceinline__ TcqParams tcq_params(int tier) {
    switch (tier) {
        case kTurbo3Tcq: return {3, 9, 512, 64, 6, 49};
        case kTurbo2Tcq: return {2, 8, 256, 64, 6, 33};
        default: return {1, 8, 256, 128, 7, 17};
    }
}

// Shared pairwise-then-shuffle sum of 128 values (the tree order the host oracle replays). `scratch` is a
// 256-float shared array; every thread participates.
__device__ __forceinline__ float block_sum128(float mine, float* scratch) {
    const int sid = threadIdx.x;
    scratch[sid] = (sid < 128) ? mine : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) scratch[sid] += scratch[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = scratch[sid];
        v += __shfl_down_sync(0xffffffffu, v, 16);
        v += __shfl_down_sync(0xffffffffu, v, 8);
        v += __shfl_down_sync(0xffffffffu, v, 4);
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (sid == 0) scratch[0] = v;
    }
    __syncthreads();
    return scratch[0];
}

// The unified append kernel. step != nullptr: decode/verify form (positions from the step buffer).
// step == nullptr: prompt form (pos = pos0 + t, src from the [T, heads, dim] K/V arrays, stage written too).
// TIER selects the codec; TCQ tiers use the 256-thread Viterbi body, vanilla tiers the thread-0 body.
template <int TIER>
__global__ void __launch_bounds__(256) kv_append_turbo_kernel(
    const float* __restrict__ kcur, const float* __restrict__ vcur, const int32_t* __restrict__ step,
    int step_stride, int planes, int64_t pos0, const int32_t* __restrict__ table, int kv_heads, int head_dim,
    int page_size, KvHostPools pools, KvHostPools host, KvHostPools stage) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const int h = blockIdx.x, g = blockIdx.y;
    const int sid = threadIdx.x;

    // Position + source row (same expression in both modes; the prompt form has step_stride = heads*dim and
    // derives pos from pos0).
    const long long pos = (step != nullptr) ? (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos)
                                            : pos0 + step_idx;
    const float* src = (is_v ? vcur : kcur) + (long long) step_idx * kv_heads * head_dim + (long long) h * head_dim +
                       g * 128;

    constexpr int kIsTcq = (TIER == kTurbo3Tcq || TIER == kTurbo2Tcq || TIER == kTurbo1Tcq);
    const TcqParams P = kIsTcq ? tcq_params(TIER) : TcqParams{0, 0, 0, 0, 0, 0};
    const int gb = (int) (kIsTcq ? P.qs_bytes + 3 : (TIER == kTurbo4 ? sizeof(block_turbo4_0) : sizeof(block_turbo8_0)));
    const int row_bytes = (head_dim / 128) * gb;

    const long long page = (long long) table[pos / page_size];
    const long long host_row = ((pos / page_size) * kv_heads + h) * (long long) page_size + (pos % page_size);
    const long long pool_row = (page * kv_heads + h) * (long long) page_size + (pos % page_size);
    const bool write_pool = page >= 0;
    const bool write_host = host.k_turbo != nullptr;
    const bool write_stage = stage.k_turbo != nullptr;
    if (!write_pool && !write_host && !write_stage) return;

    // The three destinations this cell writes (VRAM page if resident, host copy always, stage if given).
    uint8_t* dsts[3];
    long long rows[3];
    int ndst = 0;
    if (write_pool)  { dsts[ndst] = is_v ? pools.v_turbo : pools.k_turbo;   rows[ndst] = pool_row; ndst++; }
    if (write_host)  { dsts[ndst] = is_v ? host.v_turbo : host.k_turbo;     rows[ndst] = host_row; ndst++; }
    if (write_stage) { dsts[ndst] = is_v ? stage.v_turbo : stage.k_turbo;   rows[ndst] = host_row; ndst++; }

    if (!kIsTcq) {
        if (sid != 0) return;
        for (int d = 0; d < ndst; d++)
            turbo_quantize_vanilla_group<TIER>(src, dsts[d] + rows[d] * row_bytes + g * gb);
        return;
    }

    // ---- TCQ body ----
    __shared__ float x[128];
    __shared__ float cost[2][512];
    __shared__ float pred_min[128];
    __shared__ uint8_t bt[128 * 128];
    __shared__ float warp_min_cost[8];
    __shared__ int warp_min_idx[8];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = src[sid];
    __syncthreads();

    const float sq = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    const float grp_norm = sqrtf(block_sum128(sq, cost[0]));
    const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // Forward rotation (warp-shuffle form; same stage sequence as the register form -> bitwise equal).
    if (sid < 128) {
        float v = x[sid] * c_turbo_s1[sid];
        const int lane = sid & 31;
#pragma unroll
        for (int hh = 1; hh < 32; hh <<= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, v, hh);
            v = (lane & hh) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    if (sid < 128) x[sid] *= kInvSqrt128 * c_turbo_s2[sid];
    __syncthreads();

    // Viterbi forward pass.
    const float* cb = turbo_tcq_cb(TIER, is_v ? 1 : 0);
    const int low_mask = P.bt_groups - 1;
    for (int s = sid; s < P.states; s += 256) cost[0][s] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        float* cost_rd = (t & 1) ? cost[1] : cost[0];
        float* cost_wr = (t & 1) ? cost[0] : cost[1];
        const float xt = x[t];
        if (sid < P.bt_groups) {
            const int base_prev = sid << P.k;
            float best = cost_rd[base_prev];
            int best_p = 0;
#pragma unroll
            for (int p = 1; p < (1 << P.k); p++) {   // 2^k predecessors (k=1: 2, k=2: 4, k=3: 8)
                const float c = cost_rd[base_prev | p];
                if (c < best) { best = c; best_p = p; }
            }
            pred_min[sid] = best;
            bt[t * P.bt_groups + sid] = (uint8_t) best_p;
        }
        __syncthreads();
        for (int s = sid; s < P.states; s += 256) {
            float dist = xt - cb[s];
            cost_wr[s] = pred_min[s & low_mask] + dist * dist;
        }
        __syncthreads();
    }
    // 128 steps (even): final costs in cost[0].

    // Final argmin over the states; ties -> lowest index.
    {
        float my_cost = 3.4028234663852886e38f;
        int my_idx = 0;
        for (int s = sid; s < P.states; s += 256) {
            const float c = cost[0][s];
            if (c < my_cost) { my_cost = c; my_idx = s; }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, my_cost, offset);
            const int oi = __shfl_xor_sync(0xffffffffu, my_idx, offset);
            if (oc < my_cost) { my_cost = oc; my_idx = oi; }
        }
        if (sid % 32 == 0) { warp_min_cost[sid / 32] = my_cost; warp_min_idx[sid / 32] = my_idx; }
    }
    __syncthreads();
    if (sid < 32) {
        float best = (sid < 8) ? warp_min_cost[sid] : 3.4028234663852886e38f;
        int best_idx = (sid < 8) ? warp_min_idx[sid] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float oc = __shfl_down_sync(0xffffffffu, best, offset);
            const int oi = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (oc < best) { best = oc; best_idx = oi; }
        }
        if (sid == 0) shared_initial_state = best_idx;
    }
    __syncthreads();

    // Backtrack (sequential on thread 0): emit k output bits per step.
    uint8_t* outputs = (uint8_t*) x;  // x is dead after the encode; buun reuses it the same way
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> (P.L - P.k));
            const int p = bt[t * P.bt_groups + (state & low_mask)];
            state = ((state & low_mask) << P.k) | p;
        }
        shared_initial_state = state;  // the initial state the best path started from
    }
    __syncthreads();

    // Reconstruction norm over the chosen path. State at step t = the last L/k outputs, newest in the HIGH
    // bits: state_t = OR_b out_{t-b} << ((L-k) - k*b). Steps before L/k-1 walk from the initial state.
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        const int hist = P.L / P.k;  // 3 / 4 / 8 outputs fill the state
        int cur_state;
        if (sid < hist - 1) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> P.k) | (((int) outputs[t]) << (P.L - P.k));
        } else {
            cur_state = 0;
            for (int b = 0; b < hist; b++)
                cur_state |= ((int) outputs[sid - b] & ((1 << P.k) - 1)) << ((P.L - P.k) - P.k * b);
        }
        const float c = cb[cur_state];
        my_recon_sq = c * c;
    }
    const float recon_norm = sqrtf(block_sum128(my_recon_sq, cost[0]));
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    if (TIER != kTurbo1Tcq)   // buun's 1_tcq encoder applies no alpha (turbo-quant-cuda.cuh:2773)
        corrected_norm *= is_v ? kTurboTcqNormAlphaV : kTurboTcqNormAlphaK;

    // Bitpack: prefix_bits of the initial state, then 128 k-bit outputs, LSB-first. Thread sid writes byte
    // sid; thread 0 also writes the norm. Written to every destination this cell has (dsts[] above).
    if (sid < P.qs_bytes) {
        const int init_bits = (shared_initial_state >> P.k) & ((1 << P.prefix_bits) - 1);
        uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int p = sid * 8 + bit;
            int v = 0;
            if (p < P.prefix_bits) {
                v = (init_bits >> p) & 1;
            } else {
                const int sym_bit_pos = p - P.prefix_bits;
                const int sym_idx = sym_bit_pos / P.k;
                if (sym_idx < 128) v = (outputs[sym_idx] >> (sym_bit_pos % P.k)) & 1;
            }
            packed |= (uint8_t)(v << bit);
        }
        for (int d = 0; d < ndst; d++) {
            uint8_t* base = dsts[d] + rows[d] * row_bytes + g * gb;
            base[2 + sid] = packed;
            if (sid == 0) {
                const uint16_t nbits = f16_from_f32(corrected_norm);
                *reinterpret_cast<uint16_t*>(base) = nbits;
                base[2 + P.qs_bytes] = 0;  // pad byte, deterministic (buun leaves it uninitialized; zero it
                                           // so snapshot round-trips compare clean)
            }
        }
    }
}

// ---- gather (dequant into fp16 scratch, STILL ROTATED - same contract as kv_gather_q4_step: the fallback
// attention consumes rotated Q and the output is inverse-rotated by the caller) ----
// One warp per (id, head, group, plane): 4 values per lane, shuffle butterflies for the FWHT.
__global__ void kv_gather_turbo_kernel(const uint8_t* __restrict__ k_pool, const uint8_t* __restrict__ v_pool,
                                       const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                       const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                       int tier, uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int groups_per_head = head_dim / 128;
    const long long total = n_ids * kv_heads * (long long) groups_per_head * 2;  // *2: K and V planes
    const long long work = (long long) blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (work >= total) return;
    const int lane = threadIdx.x & 31;

    const bool is_v = (work & 1) != 0;
    const long long w2 = work >> 1;
    const int g = (int) (w2 % groups_per_head);
    const int h = (int) ((w2 / groups_per_head) % kv_heads);
    const long long id = w2 / (groups_per_head * kv_heads);

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * (long long) page_size + (cell % page_size);

    const int gb = (int) kv_turbo_block_bytes(tier);
    const int row_bytes = groups_per_head * gb;
    const uint8_t* blk = (is_v ? v_pool : k_pool) + row * row_bytes + g * gb;
    const float norm = f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
    const uint8_t* qs = blk + 2;

    // Decode 4 values: lane owns elements {lane, lane+32, lane+64, lane+96}. Output stays in the ROTATED
    // domain (the dot products are rotation-invariant; the caller inverse-rotates the attention output).
    float v[4];
    const float alpha = turbo_tcq_decode_alpha(tier, is_v ? 1 : 0);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int j = lane + 32 * i;
        if (tier == kTurbo8) {
            v[i] = c_turbo8_centroids[qs[j]] * norm;
        } else if (tier == kTurbo4) {
            const uint8_t idx = (j & 1) ? (qs[j / 2] >> 4) : (qs[j / 2] & 0xF);
            v[i] = c_turbo4_centroids[idx] * norm;
        } else {
            const int st = turbo_tcq_read_state(qs, j, tier == kTurbo3Tcq ? 3 : (tier == kTurbo2Tcq ? 2 : 1));
            v[i] = turbo_tcq_cb(tier, is_v ? 1 : 0)[st] * norm * alpha;
        }
    }

    uint16_t* dst = (is_v ? v_scratch : k_scratch) + ((id * kv_heads + h) * (long long) head_dim) + g * 128;
#pragma unroll
    for (int i = 0; i < 4; i++) dst[lane + 32 * i] = f16_from_f32(v[i]);
}

void need_turbo_geom(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (two FWHT-128 rotation groups)\n", what);
        std::exit(1);
    }
}

template <int TIER>
void launch_append(const float* kcur, const float* vcur, const int32_t* step, int step_stride, int n_steps,
                   int64_t pos0, const int32_t* page_table, const QsaShapes& s, void* stream,
                   const KvHostPools* pools, const KvHostPools* host, const KvHostPools* stage) {
    const int planes = (vcur == nullptr || vcur == kcur) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / 128), (unsigned) (n_steps * planes));
    const KvHostPools p = pools ? *pools : KvHostPools{}, h = host ? *host : KvHostPools{},
                             st = stage ? *stage : KvHostPools{};
    kv_append_turbo_kernel<TIER><<<grid, 256, 0, (cudaStream_t) stream>>>(
        kcur, vcur, step, step_stride, planes, pos0, page_table, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, p, h, st);
    check("kv_append_turbo launch");
}

}  // namespace

// ---- host entry points ----

void turbo_rotate_forward128_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    upload_tables();
    turbo_rotate128_kernel<<<(unsigned) n_rows, 128, 0, (cudaStream_t) stream>>>(src, dst, n_rows * 128, 0);
    check("turbo_rotate_forward launch");
}

void turbo_rotate_inverse128_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    upload_tables();
    turbo_rotate128_kernel<<<(unsigned) n_rows, 128, 0, (cudaStream_t) stream>>>(src, dst, n_rows * 128, 1);
    check("turbo_rotate_inverse launch");
}

void kv_append_turbo_step(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, int tier, void* stream,
                          const KvHostPools* host) {
    kv_append_turbo_steps(k_turbo, v_turbo, page_table, step, 0, 1, kcur, vcur, s, tier, stream, host);
}

void kv_append_turbo_steps(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           int tier, void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_turbo_geom(s, "kv_append_turbo");
    upload_tables();
    KvHostPools pools{};
    pools.k_turbo = k_turbo; pools.v_turbo = v_turbo;
    switch (tier) {
        case kTurbo8: launch_append<kTurbo8>(kcur, vcur, step, step_stride, n_steps, 0, page_table, s, stream, &pools, host, nullptr); break;
        case kTurbo4: launch_append<kTurbo4>(kcur, vcur, step, step_stride, n_steps, 0, page_table, s, stream, &pools, host, nullptr); break;
        case kTurbo3Tcq: launch_append<kTurbo3Tcq>(kcur, vcur, step, step_stride, n_steps, 0, page_table, s, stream, &pools, host, nullptr); break;
        case kTurbo2Tcq: launch_append<kTurbo2Tcq>(kcur, vcur, step, step_stride, n_steps, 0, page_table, s, stream, &pools, host, nullptr); break;
        default: launch_append<kTurbo1Tcq>(kcur, vcur, step, step_stride, n_steps, 0, page_table, s, stream, &pools, host, nullptr); break;
    }
}

void kv_append_turbo(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, int64_t pos0, int64_t T,
                     const float* K, const float* V, const QsaShapes& s, int tier, void* stream,
                     const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_turbo_geom(s, "kv_append_turbo");
    upload_tables();
    // Prompt form: step == nullptr; the kernel derives pos = pos0 + t and reads the [T, heads, dim] arrays.
    KvHostPools pools{};
    pools.k_turbo = k_turbo; pools.v_turbo = v_turbo;
    switch (tier) {
        case kTurbo8: launch_append<kTurbo8>(K, V, nullptr, 0, (int) T, pos0, page_table, s, stream, &pools, host, stage); break;
        case kTurbo4: launch_append<kTurbo4>(K, V, nullptr, 0, (int) T, pos0, page_table, s, stream, &pools, host, stage); break;
        case kTurbo3Tcq: launch_append<kTurbo3Tcq>(K, V, nullptr, 0, (int) T, pos0, page_table, s, stream, &pools, host, stage); break;
        case kTurbo2Tcq: launch_append<kTurbo2Tcq>(K, V, nullptr, 0, (int) T, pos0, page_table, s, stream, &pools, host, stage); break;
        default: launch_append<kTurbo1Tcq>(K, V, nullptr, 0, (int) T, pos0, page_table, s, stream, &pools, host, stage); break;
    }
}

void kv_gather_turbo_step(const uint8_t* k_turbo, const uint8_t* v_turbo, const int32_t* page_table,
                          const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s, int tier,
                          uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    upload_tables();
    const int64_t total = max_ids * s.n_head_kv * (s.head_dim / 128) * 2;
    const int warps_per_block = 8;
    const unsigned blocks = (unsigned) ((total + warps_per_block - 1) / warps_per_block);
    kv_gather_turbo_kernel<<<blocks, 32 * warps_per_block, 0, (cudaStream_t) stream>>>(
        k_turbo, v_turbo, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, tier,
        k_scratch, v_scratch);
    check("kv_gather_turbo launch");
}

}  // namespace strata::kernels
