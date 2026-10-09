// src/kernels/kv_turbo_parity.cpp - Parity test for the TurboQuant KV storage (kv_turbo.hpp / kv_turbo.cu).
//
// Validates, in order:
// 1. Table integrity: every constant table (sign arrays, Lloyd-Max books, TCQ codebooks) hashes (FNV-1a over
//    the raw float bits) to the value computed from buun's source. A digit drifting in a codebook silently
//    corrupts every KV block, so this gate runs first and hard-fails.
// 2. The rotation kernel bitwise against a host sequential s1 -> FWHT-128 -> s2 reference (the same operation
//    order, not just the same math - the encode's norm correction depends on the exact tree).
// 3. Paged append per tier: the vanilla tiers (turbo8/turbo4) bitwise against a host quantizer; the TCQ tiers
//    against a host Viterbi oracle (path cost within float-rounding of optimal, stored norm self-consistent,
//    bitstream decodes to the same values the gather returns).
// 4. Gather: the fp16 scratch equals the host decode of the stored bitstream, and the K/V side-split books
//    are actually in use (the two sides decode differently on TCQ tiers).
// 5. Dot-product invariance under the rotation: <Rq, Rk> == <q, k>.
//
// Run: ctest -R kv_turbo_parity (needs a GPU; the engine's own build flags apply).

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_turbo.hpp"
#include "strata/kernels/kv_turbo_alpha.hpp"
#include "strata/kernels/kv_turbo_tables.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// FNV-1a over the raw float bits (little-endian), the same fold the checksums below were computed with.
uint64_t fnv1a_floats(const float* v, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bits = std::bit_cast<uint32_t>(v[i]);
        for (int b = 0; b < 4; ++b) {
            h ^= (uint8_t) (bits >> (8 * b));
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

// ---- host references (operation order mirrors kv_turbo.cu exactly) ----

void fwht128_host(float* x) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    for (int i = 0; i < 128; i++) x[i] *= 0.08838834764831845f;
}

void rotate_forward_host(const float* src, float* dst) {
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = src[i] * k::kTurboWhtS1[i];
    fwht128_host(x);
    for (int i = 0; i < 128; i++) x[i] *= k::kTurboWhtS2[i];
    std::memcpy(dst, x, sizeof(x));
}

uint8_t nearest4_host(float val) {  // buun's binary search over the 15 midpoints
    if (val < k::kTurbo4Mids[7]) {
        if (val < k::kTurbo4Mids[3]) {
            if (val < k::kTurbo4Mids[1]) return val < k::kTurbo4Mids[0] ? 0 : 1;
            return val < k::kTurbo4Mids[2] ? 2 : 3;
        }
        if (val < k::kTurbo4Mids[5]) return val < k::kTurbo4Mids[4] ? 4 : 5;
        return val < k::kTurbo4Mids[6] ? 6 : 7;
    }
    if (val < k::kTurbo4Mids[11]) {
        if (val < k::kTurbo4Mids[9]) return val < k::kTurbo4Mids[8] ? 8 : 9;
        return val < k::kTurbo4Mids[10] ? 10 : 11;
    }
    if (val < k::kTurbo4Mids[13]) return val < k::kTurbo4Mids[12] ? 12 : 13;
    return val < k::kTurbo4Mids[14] ? 14 : 15;
}

void quantize_turbo4_host(const float* src, k::block_turbo4_0* blk) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    float r[128];
    rotate_forward_host(x, r);
    for (int j = 0; j < 128; j += 2) {
        const uint8_t i0 = nearest4_host(r[j]), i1 = nearest4_host(r[j + 1]);
        blk->qs[j / 2] = (uint8_t) ((i1 << 4) | i0);
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        const uint8_t idx = (j & 1) ? (blk->qs[j / 2] >> 4) : (blk->qs[j / 2] & 0xF);
        const float c = k::kTurbo4Centroids[idx];
        recon_sq += c * c;
    }
    const float rn = sqrtf(recon_sq);
    blk->norm = k::f16_from_f32((rn > 1e-10f) ? norm / rn : norm);
}

void quantize_turbo8_host(const float* src, k::block_turbo8_0* blk) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    float r[128];
    rotate_forward_host(x, r);
    float absmax = 0.0f;
    for (int j = 0; j < 128; j++) absmax = std::fmax(absmax, std::fabs(r[j]));
    const float scale = absmax > 1e-10f ? absmax : 1e-10f;
    const float inv_scale = 1.0f / scale;
    for (int j = 0; j < 128; j++) {
        int idx = (int) std::lrint(r[j] * inv_scale * 127.5f + 127.5f);
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        blk->qs[j] = (uint8_t) idx;
    }
    blk->norm = k::f16_from_f32(norm * scale);
}

// TCQ bitstream decode (the read the attention path performs; kv_turbo.cu's turbo_tcq_read_state).
int tcq_read_state_host(const uint8_t* qs, int t, int kbits) {
    const int bit_pos = t * kbits;
    const int byte_idx = bit_pos >> 3, bit_off = bit_pos & 7;
    uint32_t raw = (uint32_t) qs[byte_idx] | ((uint32_t) qs[byte_idx + 1] << 8);
    if (kbits == 3) raw |= (uint32_t) qs[byte_idx + 2] << 16;
    return (int) ((raw >> bit_off) & ((1u << (kbits == 3 ? 9 : 8)) - 1u));
}

const float* tcq_cb_host(int tier, int is_v) {
    if (tier == k::kTurbo3Tcq) return is_v ? k::kTurbo3TcqCodebookV : k::kTurbo3TcqCodebook;
    if (tier == k::kTurbo2Tcq) return is_v ? k::kTurbo2TcqCodebookV : k::kTurbo2TcqCodebook;
    return is_v ? k::kTurbo1TcqCodebookV : k::kTurbo1TcqCodebook;
}

float tcq_decode_alpha_host(int tier, int is_v) {
    if (!is_v) return 1.0f;
    return tier == k::kTurbo3Tcq ? k::kTurboTcqAlphaVT3 : (tier == k::kTurbo2Tcq ? k::kTurboTcqAlphaVT2
                                                                                  : k::kTurboTcqAlphaVT1);
}

// Host Viterbi over the same rotated vector the kernel encodes (double precision: the oracle only has to
// be near-optimal, not bit-identical - the GPU path is checked to be within float-rounding of its cost).
// Returns the optimal path cost and writes the chosen states.
double tcq_viterbi_host(const float* xr, int tier, int is_v, std::vector<int>* states_out) {
    const int kb = tier == k::kTurbo3Tcq ? 3 : (tier == k::kTurbo2Tcq ? 2 : 1);
    const int L = tier == k::kTurbo3Tcq ? 9 : 8;
    const int S = 1 << L;
    const float* cb = tcq_cb_host(tier, is_v);
    std::vector<double> cost(S, 0.0), ncost(S);
    std::vector<uint8_t> bt((size_t) 128 * S);
    for (int t = 0; t < 128; t++) {
        const int low_mask = (1 << (L - kb)) - 1;
        for (int s = 0; s < S; s++) {
            // predecessors of s under ns = (prev >> kb) | (out << (L-kb)): the 2^kb states whose
            // shift lands on s - s's LOW (L-kb) bits, shifted back up, plus each p
            const int base = (s & low_mask) << kb;
            double best = 1e300; int bp = 0;
            for (int p = 0; p < (1 << kb); p++) {
                const double c = cost[base | p];
                if (c < best) { best = c; bp = p; }
            }
            const double d = (double) xr[t] - (double) cb[s];
            ncost[s] = best + d * d;
            bt[(size_t) t * S + s] = (uint8_t) bp;
        }
        std::swap(cost, ncost);
    }
    double best = 1e300; int bs = 0;
    for (int s = 0; s < S; s++) if (cost[s] < best) { best = cost[s]; bs = s; }
    if (states_out) {
        states_out->resize(128);
        int s = bs;
        for (int t = 127; t >= 0; t--) {
            (*states_out)[t] = s;
            const int p = bt[(size_t) t * S + s];
            s = ((s & ((1 << (L - kb)) - 1)) << kb) | p;
        }
    }
    return best;
}

}  // namespace

int main() {
    std::printf("=== Running kv_turbo_parity test ===\n");
    std::mt19937 rng(20261009u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // -------------------------------------------------------------
    // Test 1: table integrity (FNV-1a over float bits, computed from buun's source)
    // -------------------------------------------------------------
    std::printf("[1/6] Checking constant tables against buun's checksums...\n");
    {
        const struct { const char* name; uint64_t expect; const float* data; size_t n; } tabs[] = {
            {"s1",   0x2b87b41406b05b25ull, k::kTurboWhtS1,          128},
            {"s2",   0x7fdc270b0b0d9025ull, k::kTurboWhtS2,          128},
            {"c4",   0x2c12ebb21bb670b9ull, k::kTurbo4Centroids,     16},
            {"m4",   0xade75c7f99c53719ull, k::kTurbo4Mids,          15},
            {"c8",   0x3d41b19fe7e3a7d1ull, k::kTurbo8Centroids,     256},
            {"cb3",  0xac7446dbbbed000aull, k::kTurbo3TcqCodebook,   512},
            {"cb3v", 0x02d7b8184f09e242ull, k::kTurbo3TcqCodebookV,  512},
            {"cb2",  0x0a80ad803db190faull, k::kTurbo2TcqCodebook,   256},
            {"cb2v", 0x62e7c10be5a837ceull, k::kTurbo2TcqCodebookV,  256},
            {"cb1",  0xce2c28d1abe19bc1ull, k::kTurbo1TcqCodebook,   256},
            {"cb1v", 0x6f3e28a201624cc9ull, k::kTurbo1TcqCodebookV,  256},
        };
        for (const auto& t : tabs) {
            const uint64_t got = fnv1a_floats(t.data, t.n);
            if (got != t.expect) {
                std::fprintf(stderr, "FAIL: table %s hash %016llx != buun's %016llx\n", t.name,
                             (unsigned long long) got, (unsigned long long) t.expect);
                ++g_fail;
            }
        }
        if (g_fail == 0) std::printf("  -> All 11 tables match buun's checksums (OK)\n");
        else return 1;   // nothing below means anything with drifted tables
    }

    // -------------------------------------------------------------
    // Test 2: rotation kernel bitwise vs host sequential reference
    // -------------------------------------------------------------
    std::printf("[2/6] Verifying FWHT-128 rotation kernel vs exact host reference...\n");
    {
        const int rows = 256;
        std::vector<float> h_src(rows * 128), h_ref(rows * 128);
        for (auto& x : h_src) x = nd(rng) * 3.0f;
        for (int r = 0; r < rows; ++r) rotate_forward_host(h_src.data() + r * 128, h_ref.data() + r * 128);

        float *d_src = dalloc<float>(rows * 128), *d_dst = dalloc<float>(rows * 128);
        ck(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy src");
        k::turbo_rotate_forward128_cuda(d_src, d_dst, rows, nullptr);
        ck(cudaDeviceSynchronize(), "rotate forward");
        std::vector<float> h_dst(rows * 128);
        ck(cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy dst");
        long bad = 0;
        for (int i = 0; i < rows * 128; ++i)
            if (std::bit_cast<uint32_t>(h_dst[i]) != std::bit_cast<uint32_t>(h_ref[i])) ++bad;
        if (bad) {
            std::fprintf(stderr, "FAIL: %ld rotated values differ bitwise from the host reference\n", bad);
            ++g_fail;
        } else std::printf("  -> Rotation bitwise equal to host reference (OK)\n");

        // inverse undoes forward (to fp32 rounding)
        k::turbo_rotate_inverse128_cuda(d_dst, d_dst, rows, nullptr);
        ck(cudaDeviceSynchronize(), "rotate inverse");
        ck(cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy dst2");
        double worst = 0.0;
        for (int i = 0; i < rows * 128; ++i)
            worst = std::max(worst, (double) std::fabs(h_dst[i] - h_src[i]));
        if (worst > 1e-4) {
            std::fprintf(stderr, "FAIL: inverse rotation roundtrip error %e\n", worst);
            ++g_fail;
        } else std::printf("  -> Inverse rotation roundtrip max error %e (OK)\n", worst);
        cudaFree(d_src); cudaFree(d_dst);
    }

    // -------------------------------------------------------------
    // Test 3: paged append per tier vs host references
    // -------------------------------------------------------------
    std::printf("[3/6] Verifying paged append per tier against host references...\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv;   // 2
    const int D = (int) s.head_dim;    // 256
    const int P = (int) s.page_size;
    const int pages = 8;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 25;
    for (int i = 0; i < n_fill; ++i) {
        const int pos = positions[i];
        host_k[pos].resize(H * D);
        host_v[pos].resize(H * D);
        for (auto& x : host_k[pos]) x = nd(rng) * 2.0f;
        for (auto& x : host_v[pos]) x = nd(rng) * 2.0f;
    }

    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    const int tiers[] = {k::kTurbo8, k::kTurbo4, k::kTurbo3Tcq, k::kTurbo2Tcq, k::kTurbo1Tcq};
    const char* tier_names[] = {"turbo8", "turbo4", "turbo3_tcq", "turbo2_tcq", "turbo1_tcq"};

    for (int ti = 0; ti < 5; ++ti) {
        const int tier = tiers[ti];
        const int gb = (int) k::kv_turbo_block_bytes(tier);
        const int row_bytes = (D / 128) * gb;
        const size_t pool_bytes = (size_t) pages * H * P * row_bytes;
        uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
        uint8_t* d_v = dalloc<uint8_t>(pool_bytes);

        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
            ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
            ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
            ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");
            k::kv_append_turbo_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, tier, nullptr, nullptr);
            ck(cudaDeviceSynchronize(), "kv_append_turbo_step");
        }

        std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
        ck(cudaMemcpy(h_k.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k");
        ck(cudaMemcpy(h_v.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v");

        long bad = 0;
        double worst_cost_excess = 0.0, worst_norm_rel = 0.0;
        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            for (int is_v = 0; is_v < 2; ++is_v) {
                for (int h = 0; h < H; ++h) {
                    const long long page = (long long) table[pos / P];
                    const long long row = (page * H + h) * P + (pos % P);
                    const uint8_t* gpu = (is_v ? h_v.data() : h_k.data()) + row * row_bytes;
                    const float* src = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;

                    for (int g = 0; g < D / 128; ++g) {
                        const float* sg = src + g * 128;
                        const uint8_t* blk = gpu + g * gb;
                        if (tier == k::kTurbo4) {
                            k::block_turbo4_0 ref;
                            quantize_turbo4_host(sg, &ref);
                            if (std::memcmp(&ref, blk, sizeof(ref)) != 0) ++bad;
                        } else if (tier == k::kTurbo8) {
                            k::block_turbo8_0 ref;
                            quantize_turbo8_host(sg, &ref);
                            if (std::memcmp(&ref, blk, sizeof(ref)) != 0) ++bad;
                        } else {
                            // TCQ: rotate on the host, run the oracle, compare path cost + norm self-consistency
                            float xn[128], xr[128];
                            float norm_sq = 0.0f;
                            for (int j = 0; j < 128; j++) norm_sq += sg[j] * sg[j];
                            const float grp_norm = sqrtf(norm_sq);
                            const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
                            for (int j = 0; j < 128; j++) xn[j] = sg[j] * inv_norm;
                            rotate_forward_host(xn, xr);
                            std::vector<int> states;
                            const double opt = tcq_viterbi_host(xr, tier, is_v, &states);
                            // the GPU's path cost, replayed from its stored bitstream
                            const int kb = tier == k::kTurbo3Tcq ? 3 : (tier == k::kTurbo2Tcq ? 2 : 1);
                            const float* cb = tcq_cb_host(tier, is_v);
                            double gpu_cost = 0.0, recon_sq = 0.0;
                            for (int t = 0; t < 128; t++) {
                                const int st = tcq_read_state_host(blk + 2, t, kb);
                                const double d = (double) xr[t] - (double) cb[st];
                                gpu_cost += d * d;
                                recon_sq += (double) cb[st] * (double) cb[st];
                            }
                            const double excess = gpu_cost - opt;
                            worst_cost_excess = std::max(worst_cost_excess, excess / (opt + 1e-9));
                            if (excess > 1e-3 * (opt + 1.0)) ++bad;
                            // stored norm == grp_norm / ||recon|| * encode alpha (fp16 rounding)
                            double alpha_enc = 1.0;
                            if (tier != k::kTurbo1Tcq && is_v) alpha_enc = k::kTurboTcqNormAlphaV;
                            const double want = (double) grp_norm / std::sqrt(recon_sq) * alpha_enc;
                            const double got = k::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
                            worst_norm_rel = std::max(worst_norm_rel, std::fabs(got - want) / (std::fabs(want) + 1e-9));
                            if (std::fabs(got - want) > 1e-2 * std::fabs(want) + 1e-6) ++bad;
                        }
                    }
                }
            }
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: %s - %ld block checks failed\n", tier_names[ti], bad);
            ++g_fail;
        } else if (k::kv_turbo_is_tcq(tier)) {
            std::printf("  -> %s: paths within float-rounding of optimal (max relative excess %.2e), norms "
                        "self-consistent (max rel err %.2e) (OK)\n", tier_names[ti], worst_cost_excess,
                        worst_norm_rel);
        } else {
            std::printf("  -> %s: all pool blocks bitwise equal to host quantizer (OK)\n", tier_names[ti]);
        }
        cudaFree(d_k); cudaFree(d_v);
    }

    // -------------------------------------------------------------
    // Test 4: gather decodes exactly what the bitstream says; side-split books in use
    // -------------------------------------------------------------
    std::printf("[4/6] Verifying gather against host decode of the stored blocks...\n");
    {
        const int tier = k::kTurbo2Tcq;   // a TCQ tier exercises the bitstream reader and both books
        const int gb = (int) k::kv_turbo_block_bytes(tier);
        const int row_bytes = (D / 128) * gb;
        const size_t pool_bytes = (size_t) pages * H * P * row_bytes;
        uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
        uint8_t* d_v = dalloc<uint8_t>(pool_bytes);
        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
            ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
            ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
            ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");
            k::kv_append_turbo_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, tier, nullptr, nullptr);
            ck(cudaDeviceSynchronize(), "append");
        }
        std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
        ck(cudaMemcpy(h_k.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "d2h k");
        ck(cudaMemcpy(h_v.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "d2h v");

        const int max_ids = 128;
        int32_t* d_ids = dalloc<int32_t>(max_ids);
        uint16_t* d_ks = dalloc<uint16_t>((size_t) max_ids * H * D);
        uint16_t* d_vs = dalloc<uint16_t>((size_t) max_ids * H * D);
        std::vector<int32_t> ids(max_ids);
        for (int i = 0; i < max_ids; ++i) ids[i] = positions[rng() % n_fill];
        ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        k::kv_gather_turbo_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, tier, d_ks, d_vs, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> h_ks((size_t) max_ids * H * D), h_vs((size_t) max_ids * H * D);
        ck(cudaMemcpy(h_ks.data(), d_ks, h_ks.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h ks");
        ck(cudaMemcpy(h_vs.data(), d_vs, h_vs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h vs");

        long bad = 0; double max_vdiff = 0.0;
        for (int i = 0; i < max_ids; ++i) {
            const int cell = ids[i];
            for (int is_v = 0; is_v < 2; ++is_v) {
                for (int h = 0; h < H; ++h) {
                    const long long page = (long long) table[cell / P];
                    const long long row = (page * H + h) * P + (cell % P);
                    const uint8_t* gpu_row = (is_v ? h_v.data() : h_k.data()) + row * row_bytes;
                    const uint16_t* scr = (is_v ? h_vs.data() : h_ks.data()) + (i * H + h) * D;
                    for (int g = 0; g < D / 128; ++g) {
                        const uint8_t* blk = gpu_row + g * gb;
                        const float norm = k::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
                        const float* cb = tcq_cb_host(tier, is_v);
                        const float alpha = tcq_decode_alpha_host(tier, is_v);
                        for (int j = 0; j < 128; ++j) {
                            const int st = tcq_read_state_host(blk + 2, j, 2);
                            const float want = cb[st] * norm * alpha;
                            const float got = k::f32_from_f16(scr[g * 128 + j]);
                            max_vdiff = std::max(max_vdiff, (double) std::fabs(got - want));
                            if (std::fabs(got - want) > 1e-2f * (std::fabs(want) + 1.0f)) ++bad;
                        }
                    }
                }
            }
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: gather decode mismatch on %ld values (max diff %e)\n", bad, max_vdiff);
            ++g_fail;
        } else std::printf("  -> Gather equals host bitstream decode (max diff %e) (OK)\n", max_vdiff);

        // side-split books: K and V of the same source vector must decode differently (books differ)
        const float* cbk = tcq_cb_host(tier, 0), *cbv = tcq_cb_host(tier, 1);
        int same = 0;
        for (int i = 0; i < 256; ++i) same += cbk[i] == cbv[i];
        if (same > 32) {
            std::fprintf(stderr, "FAIL: K and V TCQ books are (near-)identical (%d/256 equal)\n", same);
            ++g_fail;
        } else std::printf("  -> K/V codebooks distinct (%d/256 shared values) (OK)\n", same);
        cudaFree(d_k); cudaFree(d_v); cudaFree(d_ids); cudaFree(d_ks); cudaFree(d_vs);
    }

    // -------------------------------------------------------------
    // Test 5: dot-product invariance under the rotation
    // -------------------------------------------------------------
    std::printf("[5/6] Verifying <Rq, Rk> == <q, k>...\n");
    {
        const int rows = 128;
        std::vector<float> h_in(rows * 128), h_rot(rows * 128);
        for (auto& x : h_in) x = nd(rng) * 2.0f;
        float* d_src = dalloc<float>(rows * 128);
        float* d_dst = dalloc<float>(rows * 128);
        ck(cudaMemcpy(d_src, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        k::turbo_rotate_forward128_cuda(d_src, d_dst, rows, nullptr);
        ck(cudaDeviceSynchronize(), "rotate");
        ck(cudaMemcpy(h_rot.data(), d_dst, h_rot.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
        double worst = 0.0;
        for (int r = 0; r + 1 < rows; r += 2) {
            double raw = 0.0, rot = 0.0;
            for (int i = 0; i < 128; ++i) {
                raw += (double) h_in[r * 128 + i] * h_in[(r + 1) * 128 + i];
                rot += (double) h_rot[r * 128 + i] * h_rot[(r + 1) * 128 + i];
            }
            worst = std::max(worst, std::fabs(raw - rot));
        }
        if (worst > 1e-3) {
            std::fprintf(stderr, "FAIL: rotated dot differs from original (max %e)\n", worst);
            ++g_fail;
        } else std::printf("  -> Dot product invariant under rotation (max diff %e) (OK)\n", worst);
        cudaFree(d_src); cudaFree(d_dst);
    }

    // -------------------------------------------------------------
    // Test 6: size contract
    // -------------------------------------------------------------
    std::printf("[6/6] Checking the size contract...\n");
    {
        const uint64_t want[] = {260, 132, 104, 72, 40};   // bytes per cell and head, head_dim 256
        for (int ti = 0; ti < 5; ++ti) {
            const uint64_t got = k::kv_turbo_bytes_per_head(256, tiers[ti]);
            if (got != want[ti]) {
                std::fprintf(stderr, "FAIL: %s bytes per head = %llu, want %llu\n", tier_names[ti],
                             (unsigned long long) got, (unsigned long long) want[ti]);
                ++g_fail;
            }
        }
        if (g_fail == 0) std::printf("  -> 260 / 132 / 104 / 72 / 40 (OK)\n");
    }

    cudaFree(d_table); cudaFree(d_kcur); cudaFree(d_vcur); cudaFree(d_step);

    if (g_fail == 0) {
        std::printf("=== kv_turbo_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_turbo_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
