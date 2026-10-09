// include/strata/kernels/kv_turbo_alpha.hpp - TCQ decode alphas, single source of truth.
//
// Ported verbatim from buun's ggml/src/ggml-cuda/turbo-tcq-alpha.cuh (values KLD-panel tuned there;
// the header's own comment: a desync between consumers silently runs prefill and decode at different
// alphas). In Strata the only consumers are the TCQ encoders in kv_turbo.cu (baked into the stored
// norm) and, later, any materialize path — keep it that way.
//
// Encode-side note: buun bakes alpha K=1.0 / V=1.04 into the stored norm (d_tcq_norm_alpha /
// d_tcq_norm_alpha_v at turbo-quant-cuda.cuh:1382-1383); these are the DECODE-side extras applied on
// top. Both sets are here so the pair can never drift apart.
#pragma once

namespace strata::kernels {

// Decode alphas (per tier, V-side; K-side decode is 1.0):
inline constexpr float kTurboTcqAlphaVT1 = 1.26f;
inline constexpr float kTurboTcqAlphaVT2 = 1.06f;
inline constexpr float kTurboTcqAlphaVT3 = 1.02f;

// Encode alphas baked into the stored norm (K=1.0 for every tier):
inline constexpr float kTurboTcqNormAlphaK = 1.0f;
inline constexpr float kTurboTcqNormAlphaV = 1.04f;

}  // namespace strata::kernels
