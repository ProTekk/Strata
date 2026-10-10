# 2026-10-09/10: the TurboQuant KV port's first A/B and quality gate

Branch `port/turbo-kv` (base `fb58e0d`), RTX 3090, unsloth-ud-iq4_xs pack, engine = this repo's
`build/strata`. Two receipts: the server-side A/B (`ab.jsonl`, `tools/ab_kv_turbo.py`, 3 rounds,
arms q4_0 / turbo4 / turbo2tcq) and the teacher-forced KL panel (`kl/`, `tools/kv_kl_panel.py`).

## Quality gate (kl/, 10/10) — PASS for turbo4 at q4_0 parity; turbo2tcq only as an opt-in tier

Method: the serve engine runs each text as one `GEN 1 <ids>` request with `STRATA_LOGPOS` +
`STRATA_LOGPOS_TOPK=20` and `--short-read` above the prompt length, so the whole prompt runs
through the serial verify windows (`Verifier::window_logprobs`): one row per position with the
teacher-forced logP(target), argmax, and top-20. Fixed experts (`--expert-cache 4000
--pcie-frac 0 --adapt-swaps 0 --adapt-every 0`), `--prompt-cache 0`, one engine per (text, arm).
`--dump-logits` cannot work here: a native pack never reaches the per-token dump site.

Reference: `int8` = K8V8 (INT8 codes + FP16 scale per 64 on both sides, no rotation). There is no
`k8v8` string in `--kv`. Every KL below is KL(int8 || arm). All 16 dumps validated: exact row
counts (511/1023/1023/8191), position sequence, target alignment.

| Text | Arm | top-1 vs int8 | KL(int8‖arm) | dPPL nats/tok |
|---|---|---:|---:|---:|
| code (512) | q4_0 | 93.9% | 0.0449 | +0.006 |
| code | turbo4 | 95.9% | 0.0423 | +0.016 |
| code | turbo2tcq | 88.6% | 0.1381 | −0.003 |
| doc (1024) | q4_0 | 93.0% | 0.0660 | −0.008 |
| doc | turbo4 | 92.4% | 0.0666 | −0.005 |
| doc | turbo2tcq | 86.2% | 0.1969 | +0.046 |
| chat (1024) | q4_0 | 80.4% | 0.5438 | −0.427 |
| chat | turbo4 | 79.2% | 0.5499 | +0.127 |
| chat | turbo2tcq | 69.9% | 0.9131 | −0.033 |
| doc8k | q4_0 | 91.7% | 0.0832 | +0.002 |
| doc8k | turbo4 | 91.6% | 0.0894 | +0.003 |
| doc8k | turbo2tcq | 84.8% | 0.2502 | +0.043 |

Reading: doc8k is the representative case — turbo4 sits at q4_0 parity (KL 0.089 vs 0.083, dPPL
+0.003) at 4.125 bpv vs 4.5. turbo2tcq is measurably worse (KL 0.14–0.25, top-1 −7 points): an
opt-in memory tier, not a default. The chat row is not a result to quote: int8 is the only arm
without rotation and the repeated-turn chat text is near-tie-dense, so the reference itself is
weakest there (q4_0 "beats" it by 0.43 nats). Note this port skips the mean-subtraction tap (no
calibration for this model); buun's f16-parity TCQ receipts ran with it — the known lever if
turbo2tcq ever needs to improve.

## Performance A/B (ab.jsonl, 10/09) — decode fine, prefill regressed, quality-neutral

Throughputs are the server's own timings; `prompt_n` excludes reused prefix tokens (verified
against the engine logs), so the PP rates below are fresh-read rates. Caveats: the 250K case
reached 205,922 fresh tokens (not the 262K target); `pp@32K` was a 14,952-token fresh read (its
16,384-token prefix was reused); TG deltas carry draft-acceptance variance; round 2 q4_0 @4K is
an interference outlier (36.0 tg / 478 pp).

Medians (3 rounds): TG — 4K: q4_0 47.2, turbo4 42.2 (−10.6%), turbo2tcq 44.1 (−6.6); 44K:
53.2 / 54.8 (+3.0) / 56.8 (+6.8); 206K: 52.3 / 50.9 (−2.7) / 53.1 (+1.5). PP (fresh-read): 44K:
2004 / 1336 (−33%) / 1141 (−43%); 206K: 1929 / 1367 (−29%) / 1085 (−44%). TTFT at 206K: 106.8 s
/ 150.7 s / 189.9 s. Needle 32K+128K, depths 10/50/90: 3/3 FOUND on every arm, every round.

Kernel-level receipt (`qsa_prompt_attn_parity`, passed 10/09 18:06): per-chunk at ctx 32768 —
q4_0 9.04 ms vs turbo4 15.96 (1.77x) vs turbo2_tcq 14.45 (1.60x). The regression is in the
prompt-attention kernel, not inferred. Compiler resources (cuobjdump, shipped objects): mode 5
runs 128 regs / 53.0 KB smem = 1 block/SM; mode 4 (q4_0) 80 regs / 38.0 KB = 2 blocks/SM. Mode 5
dequantizes tier blocks to FP16 rows at gather (norm folded in, ks/vs=1.0); mode 4 stages int8
codes + per-32 scales.

## Next

Profile the mode-5 kernel (occupancy vs gather-instruction cost) before optimizing; then one
change at a time (norm folding out of the gather into the ks/vs scale path is the smallest
measurable step; raw-block K staging is the occupancy play and stays separate). No occupancy
claim for norm folding — the FP16 rows do not change size.

Static SASS (cuobjdump -sass, shipped objects, 10/10): the gather dominates the instruction mix.
Per kernel: mode 4 (q4_0) 1936 instructions with 10 global loads (46-bit-field vector loads, 9
U16 scales); mode 5 turbo4 2456 with 230 loads — 64 LDG.E.U8 (one byte-load per code byte, the
unaligned 66 B stride forbids vectors) + 129 LDG.E (the per-nibble codebook lookups) + 122 F2F
(float→half rounds); mode 5 turbo2tcq 3072 (bitstream unpacking on top). The MMA/softmax body is
the same code across modes, so the delta is the gather: instruction-bound, on top of the
occupancy half from the 53 KB FP16 rows. A norm-fold alone removes the multiply and half the F2F
rounds; a static byte→half2 LUT (possible once the norm is out of the value) removes the 129
lookups and 64 byte-loads per group-pair down to one table read per byte. ncu still pending for
the runtime split (stall reasons, L1 hit rate).

ncu (10/10, sudo, mode-5 launches, ctx 32768): LATENCY-BOUND AT 8.33% OCCUPANCY, limited by
shared memory (Block Limit Shared Mem = 1; registers would allow 4 blocks). One active warp per
scheduler, 81.7% of cycles with no eligible warp, one instruction issued every 5.5 cycles per
scheduler. DRAM 1.18% — nothing is bandwidth-starved; L1/TEX 46% — the codebook lookups do hit
L1 (94a8649 worked), the gather is L1-latency-bound on its dependent chain (byte-load → lookup →
F2F → STS) with four warps per SM to hide it. turbo8: 55.9% mem / 35.4% SM; turbo4: 45.7% /
44.7%; both 8.33% occupancy. q4_0 (mode 4) was not in the captured window; its 2 blocks/SM =
16.7% is still low, but its gather chain is ~10 loads long instead of ~230.

Implication for the experiment order: occupancy is the structural lever (4 warps/SM cannot hide
any chain), instruction-count reduction is the secondary one. Cheapest probe first: a per-mode
chunk of 16 cells (D1_CH is already a build knob, but it is global — a mode-5-only chunk needs a
small template change; halving the staged rows takes mode 5 to 34.9 KB = 2 blocks/SM). Chunking
reorders the online-softmax sums, so the parity gate re-runs either way. Then norm-fold +
byte→half2 LUT (shortens the chain, halves F2Fs; math change, parity re-run). Raw-block staging
(→ ~28 KB, 3 blocks/SM) stays the last, biggest step.

CH=16-for-mode-5 PROBE: NEGATIVE (10/10, reverted). Per-mode chunk (Smem::CH_ = 16 for mode 5,
34.9 KB = 2 blocks/SM, regs 128 → 72-94): parity PASSed on all tiers, but the kernel got slower -
turbo4 15.96 -> 18.02 ms/chunk (+13%), turbo2_tcq 14.45 -> 19.51 (+35%), turbo3_tcq +27%, turbo8
flat. Doubling blocks/SM does not pay when the block's own parallelism halves: at CH=16 the score
loop runs on warps 0-1 only (warps 2-3 idle), the gather leaves half the threads idle, and every
per-chunk fixed costs (syncs, softmax, alpha rescale) doubles per unit of work. So the 8.33%
occupancy ceiling is real but chunk-splitting is the wrong way into it - the occupancy play must
keep CH=32 and shrink the staged bytes instead (raw-block staging), and the chain shortening
(norm-fold + byte LUT) attacks the latency directly. Order after this probe: norm-fold + LUT
first, raw-block staging second.

NORM-FOLD + HALF LUTS: COMMITTED 8a50973 (10/10). Gather-only. The block norm (TCQ V side: also
its alpha) moved out of the per-element chain into the FP32 ks/vs scales the score and p.v paths
already multiply by; the float codebook copies became half-width tables - turbo4 as a 256-entry
half2 LUT (one aligned 32-bit read per byte yields both nibbles, no arithmetic), turbo8 and the
TCQ codebooks pre-rounded halves. Staged rows stay FP16: same 53 KB, still 1 block/SM (regs
128 -> 96-112, no occupancy claim). Parity ms/chunk (baseline -> new): turbo4 15.96 -> 15.32,
turbo2_tcq 14.45 -> 13.63, turbo3_tcq 17.04 -> 15.08, turbo1_tcq 13.94 -> 12.82, turbo8 21.63 ->
21.08. Accuracy improved where the rounding site moved (turbo4 new-vs-FP64 2.24e-05 -> 1.32e-05:
pre-rounding the codebook once + exact FP32 norm beats rounding the product per element); TCQ
deltas moved up slightly (pre-rounded codebook), all >=10x inside the bar.

Still 15.3 vs q4_0's 9.1 ms/chunk: the gather's byte loads dominate now. Within one item a
thread reads 64 consecutive bytes and the tier blocks sit at 66/130-byte strides, so each byte
load instruction touches ~32 distinct sectors across the warp - the L1 round-trip per element is
gone but the load-instruction cost is not. That is exactly what raw-block staging removes (stage
the 132-byte rows once, decode from shared); it stays the last step, and the buun cp.async
experience (66B strides burned +45-140% there) says measure before believing.

KL PANEL RE-RUN POST-8a50973 (10/10): the 16 dumps regenerated byte-identical, and that is the
correct outcome - the panel cannot see the gather. STRATA_LOGPOS forces the serial verify-window
loop (generate.cpp:9535), which runs attention through qsa_decode_attn_batch (verify.cpp:1243);
qsa_prompt_attn_batch is called only from the batched prompt path (prefill.cpp:2783). So the
table above stands as the turbo-format quality receipt (the KV encode path is untouched), and
the gather's rounding move is covered by the parity suite, where it passed with improved
accuracy. Harness note: `run` trusts existing dumps (resume-after-crash); after a binary
rebuild, clear kl/*.logpos first - the panel prints per-arm progress lines only when it runs.
