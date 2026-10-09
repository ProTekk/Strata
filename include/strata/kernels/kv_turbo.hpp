// include/strata/kernels/kv_turbo.hpp - TurboQuant KV storage formats for the QSA layers (`--kv turbo4` etc.).
//
// Ported from buun's TurboQuant (ggml/src/ggml-cuda/turbo-quant-cuda.cuh et al.). The domain contract
// (non-negotiable, from buun): K is stored ROTATED, V is stored rotated with the attention output
// inverse-rotated, and Q is PRE-ROTATED before scoring, so <Rq, Rk> = <q, k>. Per 128-value group the
// rotation is x -> x*s1 -> FWHT-128 -> x*s2 with the seed-42 signed Sylvester arrays (kv_turbo_tables.hpp).
// This is a DIFFERENT transform from the q4_0 path's FWHT-256 (kv_q4.hpp) - the two must never share code.
//
// Encode pipeline (canonical order, turbo-quant-cuda.cuh:833-955): [mean-sub: skipped, no calibration for
// this model] -> [InnerQ equalize: identity] -> L2 norm -> FWHT-128 -> quantize (nearest centroid for the
// vanilla tiers, Viterbi trellis for TCQ) -> norm correction (norm = grp_norm / ||recon||, fp16; the TCQ
// tiers bake alpha K=1.0 / V=1.04 into it). Decode = codebook_side[idx] * norm (TCQ: O(1) random access -
// state_t = read_bits(qs, t*k) - which is exactly what the sparse ids gather wants).
//
// head_dim 256 = TWO 128-value rotation groups, so a cell's K (or V) for one head is 2 blocks. Layout per
// cell and head: [group0 block][group1 block], inline norm, no separate scale arrays - simpler than both
// int8 and q4_0 (a page run is one contiguous 16-B-multiple, see kv_stream.cu's guard).
//
// Sizes per row (1 cell, 1 head, K or V), head_dim 256: turbo8_0 260 B, turbo4_0 132 B, turbo3_tcq 104 B,
// turbo2_tcq 72 B, turbo1_tcq 40 B. Quality/speed receipts: bench/results/2026-10-XX-turbo-kv.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cstdint>

namespace strata::kernels {

/// The rotation group size (head_dim 256 = 2 groups). Fixed by the FWHT-128 transform; snapshots
/// validate head_dim % 128 == 0 for turbo formats.
inline constexpr int QK_TURBO_GROUP = 128;

/// The five ported tiers, in size order. The engine exposes turbo8/turbo4/turbo3tcq/turbo2tcq/
/// turbo1tcq; turbo1_tcq is a research tier (1.25 bpv) that stays flag-off.
enum TurboTier : int { kTurbo8 = 0, kTurbo4 = 1, kTurbo3Tcq = 2, kTurbo2Tcq = 3, kTurbo1Tcq = 4 };

#pragma pack(push, 1)
// Block layouts verbatim from buun's ggml-common.h:348-462. One block = one 128-value rotation group.
struct block_turbo8_0  { uint16_t norm; uint8_t qs[128]; };  // 130 B: uniform 256-grid, absmax scale in norm
struct block_turbo4_0  { uint16_t norm; uint8_t qs[64]; };   //  66 B: 16-level Lloyd-Max, 2 codes/byte
struct block_turbo3_tcq{ uint16_t norm; uint8_t qs[49]; uint8_t pad; };  // 52 B: 390-bit trellis bitstream
struct block_turbo2_tcq{ uint16_t norm; uint8_t qs[33]; uint8_t pad; };  // 36 B: 262-bit trellis bitstream
struct block_turbo1_tcq{ uint16_t norm; uint8_t qs[17]; uint8_t pad; };  // 20 B: 135-bit trellis bitstream
#pragma pack(pop)

static_assert(sizeof(block_turbo8_0)   == 130, "block_turbo8_0 must be 130 bytes");
static_assert(sizeof(block_turbo4_0)   ==  66, "block_turbo4_0 must be 66 bytes");
static_assert(sizeof(block_turbo3_tcq) ==  52, "block_turbo3_tcq must be 52 bytes");
static_assert(sizeof(block_turbo2_tcq) ==  36, "block_turbo2_tcq must be 36 bytes");
static_assert(sizeof(block_turbo1_tcq) ==  20, "block_turbo1_tcq must be 20 bytes");

/// Bytes of one 128-value block for a tier (device-callable: the gather kernel sizes rows with it).
__host__ __device__ inline uint64_t kv_turbo_block_bytes(int tier) {
    switch (tier) {
        case kTurbo8: return sizeof(block_turbo8_0);
        case kTurbo4: return sizeof(block_turbo4_0);
        case kTurbo3Tcq: return sizeof(block_turbo3_tcq);
        case kTurbo2Tcq: return sizeof(block_turbo2_tcq);
        default: return sizeof(block_turbo1_tcq);
    }
}

/// Bytes per cell and KV head: head_dim 256 = 2 groups -> 260 / 132 / 104 / 72 / 40.
inline uint64_t kv_turbo_bytes_per_head(int head_dim, int tier) {
    return (uint64_t) (head_dim / QK_TURBO_GROUP) * kv_turbo_block_bytes(tier);
}

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_turbo_bytes_per_cell(const QsaShapes& s, int tier) {
    return (uint64_t) s.n_head_kv * kv_turbo_bytes_per_head((int) s.head_dim, tier) * 2;
}

/// TCQ tiers (the Viterbi encoders + bitstream decode); vanilla tiers quantize per 128-group directly.
inline bool kv_turbo_is_tcq(int tier) { return tier >= kTurbo3Tcq; }

/// The engine's active turbo tier (set once by `--kv turbo*` parsing in generate.cpp, read by the format
/// plumbing that only gets a `fmt` - kv_stream's runs_of / kv_block_bytes). Meaningful only when the KV
/// format is kKvTurbo.
extern int g_kv_turbo_tier;

// ---- rotation (the domain contract) ----

/// Forward rotation of rows of 128 floats: x*s1 -> FWHT-128 (orthonormal, 1/sqrt(128)) -> *s2.
/// `qcur`/K/V rows are rotated with this before the quantized append; Q is rotated with it before scoring.
void turbo_rotate_forward128_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
/// Inverse: x*s2 -> FWHT-128 -> *s1 (the sign arrays swap; FWHT is self-inverse). Used on the attention
/// output (a mix of rotated values) and by the gather/materialize paths.
void turbo_rotate_inverse128_cuda(const float* src, float* dst, int64_t n_rows, void* stream);

// ---- write path (mirrors kv_q4.hpp's trio; host copy always, VRAM page only if resident) ----

/// Append the (raw, unrotated) cell at step[kStepPos]; the kernel does the rotation.
void kv_append_turbo_step(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, int tier, void* stream,
                          const KvHostPools* host = nullptr);

/// The verify window: `n_steps` cells from the step buffer (stride `step_stride`, like q8's batched form).
void kv_append_turbo_steps(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           int tier, void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout) when given.
void kv_append_turbo(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, int64_t pos0, int64_t T,
                     const float* K, const float* V, const QsaShapes& s, int tier, void* stream,
                     const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim], dequantized and
/// INVERSE-rotated back to the original domain (the fallback-attention slow path; the fused read path in
/// qsa_decode_attn reads the blocks directly and never materializes).
void kv_gather_turbo_step(const uint8_t* k_turbo, const uint8_t* v_turbo, const int32_t* page_table,
                          const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s, int tier,
                          uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
