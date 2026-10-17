# 2026-10-16: the port rebased onto v0.1.44.1, a fresh baseline, and the encode discovery

Branch `port/turbo-kv`, RTX 3090, unsloth-ud-iq4_xs pack, engine = this repo's `build/strata`.
Continues `bench/results/2026-10-09-turbo-kv/README.md` (the port's first session). Receipts here:
`ab.jsonl` (2-round baseline on the merged engine), `ab_wide_enc32.jsonl` (1 round after the two
kernel fixes), `cfg-*.json`, `server-*.log`.

## The rebase (merge 35c88047)

origin/main had moved 622 commits past the port's base (v0.1.41-era fb58e0d -> v0.1.44.1). Merged,
not rebased: one resolution instead of 18 replays, and the repo's own convention. One content
conflict (qsa_decode_attn.cu): upstream added QUERY_SWIZZLE/REDUCE12 template params (the gfx906
int8 specialization) exactly where the port had TIER. TIER moved LAST - an int in a bool template
slot converts silently (kTurbo4=1 would bind QUERY_SWIZZLE=true). A `git checkout -m` re-merge
then dropped the port's `load8<KV_MODE, TIER>` in the warp-scoring else branch (alignment picked
the upstream copy); caught by auditing every instantiation and call site - turbo4 would have
decoded through turbo8 layouts with no compile error. Lesson recorded: after any conflict re-run,
audit every template instantiation positionally; int->bool and bool->int template bindings compile.

Audit: all 13 shared files carry the port's delta byte-identical vs origin/main; every
kv_append/gather/rotate seam has its turbo sibling; the MTP drafter stays INT8 under turbo by
design. Also dropped a dead duplicate `else if (st.kv_turbo)` branch in prefill.cpp's draft-KV
fill (an editing scar from the original port).

## Receipts on the merged tree (pre-optimization)

- kv_turbo_parity 6/6, qsa_prompt_attn_parity 12/12, all NEWHASHes bitwise identical to pre-merge.
- ctest subset (kv|qsa|turbo|snapshot|verify|prefill|mtp|gdn|stream): 20/20.
- turbo4 smoke: boots, serves, KV streaming live, drafts accepted.
- Kernel ms/chunk unchanged by the merge: turbo4 15.21, q4_0 9.06.

## Fresh A/B baseline on the merged engine (ab.jsonl, 2 rounds, alternating order)

Medians. Upstream's 0.1.42-0.1.44 decode work (route-tail-skip, pipelined decode) lifts TG for
every arm ~10-15% vs the 10/10 numbers; PP is unchanged. Round-to-round spread < 1% on PP
(turbo4 pp@250K: 1673.6 / 1673.4).

| case | q4_0 | turbo4 | turbo2tcq | t4 vs q4 | t2 vs q4 |
|---|---:|---:|---:|---:|---:|
| pp@32K | 1974.8 | 1697.8 | 1760.7 | -14.0% | -10.8% |
| pp@46080 | 2049.6 | 1785.3 | 1829.6 | -12.9% | -10.7% |
| pp@250K | 1930.0 | 1673.5 | 1728.2 | -13.3% | -10.5% |
| tg@4096 | 43.2 | 42.8 | 44.8 | -0.9% | +3.8% |
| tg@46080 | 62.8 | 64.5 | 61.8 | +2.8% | -1.6% |
| tg@250K | 55.9 | 53.1 | 63.3 | -4.9% | +13.3% |

Needle 6/6 on every arm (both rounds, 32K+128K, depths 10/50/90). TTFT@250K: 106.8 / 123.2 /
119.3 s. TG deltas carry the usual draft-acceptance variance (turbo2tcq accepted 13/13, 23/26,
18/21 both rounds; q4_0's 250K acceptance swung between rounds). The tg@4096 pp column is the
known first-prompt file-tier I/O confound (7+ GB of experts cold) - not a result to quote.

BACKLOG #1 CLOSED: the 10/10 -24.7% turbo4 pp@46080 outlier did not reproduce (-12.9%, both
rounds inside the band). Noise, not the kv_resident=32768 streaming boundary.

## Backlog #2: the PP decomposition (nsys, no sudo needed - the parity of gpu__time sums is enough)

One 32K prompt (31,336 fresh tokens) per arm under `nsys profile -t cuda`, cuda_gpu_kern_sum:

| kernel group | turbo4 (wide-load binary) | q4_0 | delta |
|---|---:|---:|---:|
| prompt_attn (mode 5 vs 4) | 2102.2 ms / 48 | 1509.2 ms / 47 | +593 ms (25% of penalty) |
| kv_append (encode) | 1652.5 ms / 48 | 90.0 ms / 106 | **+1562 ms (67%)** |
| rotation (rotate128 vs fwht256) | 142.3 ms / 96 | 47.0 ms / 200 | +95 ms (4%) |
| total GPU kernel time | 13.222 s | 10.894 s | +2.33 s |

The gather was NOT the main lever - the encode was. Root cause (cuobjdump -res-usage):
the vanilla tiers run buun's thread-per-group math as thread 0 of a 256-thread block with
`float x[128]` in registers: REG 255 + STACK 608-640 (spilled), 255x256 = the whole SM register
file -> ONE block/SM, ONE working warp, 255/256 of every block idle. The TCQ tiers (shared-memory
bodies, REG 26) never had this - which is also why turbo2tcq's PP beat turbo4's end to end.

## The two fixes on top of the merge

1. WIDE WORD LOADS in the mode-5 gather (1bcb6a5c, backlog #3): the code-byte run is read as
   floor-aligned u32 words + one uniform funnel shift per word. A 16-byte floor would need a
   runtime word offset -> dynamic register indexing -> local-memory spill; the 66/130-B strides
   keep every run at 0 or 2 mod 4, so a 4-byte floor needs none. SASS (turbo4 mode 5): LDG.E.U8
   64 -> 0, instructions 2456 -> 1848 (below q4_0 mode 4's 1936), REG 96 STACK 0 LOCAL 0, every
   tier spill-free. Pools take +16 B tails for the <=7 B window overread (layer.cpp; the elastic
   corner is documented at both sites). Parity 12/12, NEWHASHes bitwise identical. ms/chunk:
   turbo4 15.21 -> 12.77 (-16%), turbo8 20.87 -> 15.27 (-27%), t3 15.03 -> 13.67, t2 13.53 ->
   13.11, t1 12.72 -> 12.43; controls unchanged.
2. 32-THREAD VANILLA APPEND (5a8f0064): same math, same thread-0 work, 8 blocks/SM instead of 1.
   Encode 1652.5 -> 887.0 ms (-46%); traced PP rate 1194.5 -> 1692.4 (+42%). kv_turbo_parity
   bitwise, as before. The x[128] spill itself remains - the next lever is a cooperative
   warp-per-group body (x in shared, the two serial norm chains on one lane, FWHT in the
   already-proven bitwise-equal shuffle form) or buun's flat multi-group-per-block shape.

Post-fix penalty estimate (kernel sums): encode +797 ms (51%), gather +596 ms (38%), rotate
+95 ms (6%) of a ~1.56 s total at 32K.

## Confirmatory A/B (ab_wide_enc32.jsonl, 1 round, post both fixes)

| case | q4_0 | turbo4 | turbo2tcq | t4 vs q4 | t2 vs q4 |
|---|---:|---:|---:|---:|---:|
| pp@32K | 2005.3 | 1837.1 | 1763.1 | -8.4% | -12.1% |
| pp@46080 | 2096.6 | 1900.1 | 1821.4 | -9.4% | -13.1% |
| pp@250K | 1954.0 | 1789.7 | 1726.5 | -8.4% | -11.6% |

turbo4's PP penalty: -13..-14% -> **-8.4..-9.4%** (absolute: 1698->1837, 1785->1900,
1674->1790). turbo2tcq's absolute PP is flat (1761->1763, 1830->1821, 1728->1727) - expected:
the TCQ encoder was not in scope for either fix and its gather gained only ~3%; its percentage
moves only because the q4_0 control drifted +1.2..+2.3% between runs (machine state). TG this
round: turbo4 38.8/63.6/54.0 vs q4_0 44.1/67.4/58.8 with draft acceptances 12/17, 22/24, 24/36
vs 17/38, 19/24, 26/41 - single-round draft variance around the 2-round baseline's parity; the
tg@4096 pp column is the known cold file-tier confound again (333.5/426.5 - not a result).
Needle 3/3 on every arm. TTFT@250K: 105.5 / 115.2 (+9.2%, was +15.3%) / 119.4 s.

## The warp-cooperative vanilla encode (e2b45452) and its A/B (ab_coop_encode.jsonl, 1 round)

The 32-thread fix had only halved the encode because the deeper shape problem remained: one
working thread per group with x[128] in registers (REG 255 + 608-640 B spill), the whole
pipeline RE-RUN per destination (pool+host+stage = up to 3x on the streaming prompt path), and
the kernel's static smem (the TCQ body's 21.5 KB) capping it at 2-4 blocks/SM. The rewrite:
8 groups per 256-thread block, one warp per group, coalesced float4 load, the two serial norm
chains kept as buun's exact left folds on lane 0, the FWHT in the already-proven bitwise shuffle
stage order, one quantize result stored to every destination as aligned u16s. TCQ untouched.

Receipts: kv_turbo_parity bitwise on both vanilla tiers; REG 32/38, STACK 0, LOCAL 0; nsys 32K
prompt: kv_append_turbo 1652.5 -> 887.0 -> **17.5 ms** per 48 launches - the turbo encode is now
5x cheaper than q4_0's own append (90.0 ms). qsa_prompt_attn_parity 12/12, ctest subset green.

| case | q4_0 | turbo4 | turbo2tcq | t4 vs q4 | (pre-coop t4 vs q4) |
|---|---:|---:|---:|---:|---:|
| pp@32K | 2016.4 | 1918.9 | 1779.4 | **-4.8%** | -8.4% |
| pp@46080 | 1766.9 | 1683.0 | 1847.0 | **-4.7%** | -9.4% |
| pp@250K | 1963.2 | 1883.3 | 1733.0 | **-4.1%** | -8.4% |

(The 46080 case ran slow for every arm this round - absolute pp wobbles with machine state; the
same-round control is the comparison. turbo2tcq is flat within noise, as expected: the TCQ
encoder was not touched - it doubles as a second control.) TG: 47.7/64.2/56.4 vs q4_0
48.5/66.6/59.1 (-1.7/-3.6/-4.6%) with near-identical draft acceptances (19/23 vs 19/24, 24/36
vs 26/41) - the residual TG lag now points at the decode path, where qsa_decode_attn.cu still
reads its codebook tables from __constant__ memory with data-divergent indices (serialized
replays; the prompt gather escaped exactly this in 94a8649 via __device__ L1 tables). Needle
3/3 all arms; TTFT@250K 105.0 / 109.5 (+4.2%) / 118.9.

Session arc of turbo4's PP penalty vs q4_0: -13..-14% (at merge) -> -8.4..-9.4% (wide loads +
32-thread blocks) -> **-4.1..-4.8%** (cooperative encode). Remaining PP penalty is ~86% the
mode-5 prompt gather (+647 ms GPU at 32K) - raw-block staging territory.

## Raw-block staging for the turbo4 gather (5b67fbda) - mode 5 now beats q4_0

Backlog #4, the structural lever, landed for turbo4 (the priority tier; other tiers keep the FP16
staging path). Smem<5, kTurbo4> stages the pool's aligned code WORDS - 32 u32 per (side, cell) at
a 34-word bank-staggered stride - instead of dequantized FP16 rows: 64 B staged per item instead
of 256 B, dynamic smem 53.0 -> 28.5 KB, 1 -> 3 blocks/SM (smem was ncu's sole limiter; occupancy
8.33% -> 25%, REG 96 unchanged, zero spill). The half2-LUT dequant moved INTO the MMA fragment
loads, where each b register is exactly one LUT entry: the score path reads two broadcast words
per 16-dim step (all four tig lanes share a word; the LUT's half2 = dims k0+2tig,+1), the p.v
path two cells per dim with d's parity selecting the half. Dequant counts are conserved - every
dim decodes once per chunk, same as the staged form. The masked-cell zero-fill disappeared:
ks/vs = 0 already makes the score and the folded probability +-0, and the LUT maps ANY byte to
finite halves (no NaN path), so stale smem is harmless. The window tail overread rides the pools'
+16 B spare from 1bcb6a5c. The buun cp.async warning did not apply - this is LDG -> funnel shift
-> STS through registers, not cp.async at 66-B strides.

Receipts: parity 12/12 with turbo4's NEWHASHes bitwise identical; ms/chunk turbo4 12.70 -> 7.90
at ctx 32768 (-38%) and 3.42 -> 2.08 at ctx 1500 - q4_0 is 9.02, so mode 5 is now FASTER than the
control kernel; turbo8/TCQ/q4_0/int8/fp16 unchanged; kv_turbo_parity bitwise; ctest 20/20.

nsys 32K prompt (production shapes, same harness as above):

| kernel group | at merge | post-coop | post-raw-staging | q4_0 |
|---|---:|---:|---:|---:|
| prompt_attn | 2102.2 | 2156.6 | **1314.6** | 1509.2 |
| kv_append | 1652.5 | 17.5 | 17.5 | 90.0 |
| rotation | 142.3 | 144.0 | 148.8 | 47.0 |
| total GPU | 13222 | 11644 | **10877** | 10894 |

The turbo4 GPU-time penalty vs q4_0 is GONE at the kernel level (-17 ms): the attention saving
(-195) plus the encode saving (-72) now outweigh the rotation cost (+102). What remains of any
wall-clock PP gap is rotation (the bandwidth-bound FWHT-128 pairs on Q/out, backlog #7's fusion
candidate) and turbo's cheaper streaming traffic (132 vs 144 B per row) pulling the other way.

## Raw-staging A/B (ab_raw_staging.jsonl, 1 round, q4_0 + turbo4)

| case | q4_0 | turbo4 | t4 vs q4 |
|---|---:|---:|---:|
| pp@32K | 2017.5 | 2027.4 | **+0.5%** |
| pp@46080 | 1854.2 | 2125.1 | **+14.6%** (the control wobbled low this round; turbo4's 2125 is its best-ever absolute) |
| pp@250K | 1961.5 | 1969.5 | **+0.4%** |
| tg@4096 | 47.5 | 47.4 | -0.2% |
| tg@46080 | 67.1 | 69.1 | +3.0% |
| tg@250K | 58.9 | 56.7 | -3.7% (draft variance: 24/36 vs 26/41 acceptances) |

TTFT: 32K 7.41 vs 7.44; 46K 20.8 vs 23.8 (-13%); 250K 104.7 vs 105.1 - the TTFT gap went
+15.3% (at merge) -> +4.2% (coop encode) -> parity. Needle 3/3 at 32K and 128K on both arms.

**The PP penalty is gone: turbo4 runs at q4_0 speed (-0.2..+14.6%, parity within round noise on
the two reliable cases) at 4.125 bpv vs 4.5 - the KV footprint and streaming-traffic savings are
now free.** Session arc: -13..-14% -> -8.4..-9.4% (wide loads + 32-thread append) -> -4.1..-4.8%
(coop encode) -> parity (raw-block staging). Every step bitwise against the parity gates.

## The drafter goes turbo (d1dd2aaf) - Phase 2 closed, the footprint win lands

The MTP drafter's INT8 KV fallback was eating the entire main-pool saving: its ring is 1024 B/cell
vs turbo4's 528. The fallback existed because the port's Phase 1 scoped the drafter out - but its
identity-ids read IS qsa_decode_attn_batch (KV_MODE 4 covers every tier since the port); only the
mtp.cpp wiring was missing: kv_append_turbo_steps/step seams (batched + per-token, mirroring
verify.cpp) and the rotate pair in record_rest (forward128 on q, inverse128 on the attention out,
2 groups/head). prefill.cpp's draft-fill branch came alive (it was there, dead under the fallback);
snapshot/stream were already fmt-aware.

Measured (huihui-iq3_s pack, live config, 262K context, same-engine q4_0 control):

| | q4_0 (live) | turbo4 + INT8 drafter | turbo4 drafter |
|---|---:|---:|---:|
| draft layer | 978 MiB | 1098 MiB | **966 MiB** |
| expert cache free | 14.51 GiB | 14.53 GiB | **14.66 GiB** |
| expert slots | 5489 | 5499 | **5551** |
| idle VRAM | 23667 MiB | 23811 MiB | 23669 MiB |

The full accounting of "what does turbo4 buy on this rig": the model is hybrid - 12 of 48 layers
hold KV (full_attention_interval 4) - so the main pool is only 1.69 GiB (q4_0) vs 1.55 (turbo4),
a 144 MiB saving; the turbo drafter adds 12 MiB more vs q4_0's. Total +157 MiB to the expert
cache = +62 slots (~+1%). The footprint win is real but modest at 262K on a 12-KV-layer model;
it scales with max_context and matters most on KV-bound cards/configs. Functional: ctest 20/20,
parity green, smoke acceptance 20/22, coherent at 100K depth (pp 2529, unchanged).

## Drafter-turbo A/B (ab_drafter_turbo.jsonl, 1 round, q4_0 + turbo4)

| case | q4_0 | turbo4 (turbo drafter) | delta |
|---|---:|---:|---:|
| pp@32K | 2020.2 | 2043.9 | +1.2% |
| pp@46080 | 2104.7 | 2127.5 | +1.1% |
| pp@250K | 1962.7 | 1986.7 | +1.2% |
| tg@4096 | 45.6 | 55.0 | (q4_0 draft 17/38 - a bad-luck round) |
| tg@46080 | 66.8 | 60.9 | (acceptances 19/24 vs 19/26 - variance) |
| tg@250K | 58.8 | 57.7 | -1.9% |
| TTFT 32K/46K/250K | 7.43/20.96/105.04 | 7.34/20.74/103.77 | turbo4 ahead on all |

Drafter acceptance with the turbo4 KV: 0.75 / 0.73 / 0.68 - the same band as the INT8-drafter
rounds (0.71 / 0.83 / 0.67): halving the drafter KV precision cost nothing measurable. Needle
3/3 both arms both lengths. PP is now +1.1..+1.2% (turbo4 slightly AHEAD of q4_0 on every
reliable case). Final position: at 262K on this hybrid model turbo4 gives a ~157 MiB footprint
saving (+62 expert slots), PP/TTFT parity-or-better, TG parity within draft variance, and the
quality gate from 10/10 (KL at q4_0 parity) still stands - the encode path has been bitwise
unchanged since.

## Config probe: vram-reserve and pool-workers (config_probe.jsonl, 10/16 evening)

Prompted by the r/LocalLLaMA "Strata performance finetuning" thread (u/Dreeew84, 2x3060). Four
boots of the turbo4 huihui config, tg@46080 (256 tok) + tg@250K (128 tok) each:

| variant | tg@46K | tg@250K | pp@250K | #2065 warnings | slots |
|---|---:|---:|---:|---:|---:|
| A: reserve 700, workers auto(15) | 79.9 | 84.7 | 2275.6 | 1 | 5551 |
| B: reserve 896, workers auto | 81.3 | 82.7 | 2276.6 | 0 | 5474 |
| C: 896, workers 23 | 78.9 | 81.6 | 2277.4 | 0 | 5474 |
| D: 896, workers 31 | 74.4 | 80.0 | 2274.9 | 0 | 5474 |

ADOPTED: --vram-reserve-mib 896 (engine's own ask; kills the warnings; PP identical to the digit,
TG within noise; costs 77 slots ~1.4%). Applied to strata-huihui-iq3_s-turbo4.json AND the live
strata-huihui-iq3_s.json. REJECTED: --pool-workers overrides - default (15 workers + host, physical
cores minus one) is best on this rig; 31 workers is -8% at 46K. The thread's +10% pool-workers win
is a 2x3060 story (CPU-side expert work); here R4 keeps resident experts on the GPU at pcie-frac
0.00 and the CPU pool has little to do. The thread's other two knobs are multi-GPU-only in the
code: --pipeline-windows ("a layer split on exactly two GPUs", generate.cpp) and --trim-stage-
weights (guarded by `if (!multi_gpu)`). Draft acceptances identical across all four variants.
Open (needs root/GUI, handed to Jacob): the NVIDIA "CUDA Sysmem Fallback Policy" (driver 546+;
darksteelsteed's tip) - Prefer No Sysmem Fallback makes VRAM overcommit fail honestly instead of
silently paging decode into system RAM.

## Beyond turbo: full-engine decompositions (10/16 evening, nsys on the final turbo4 binary)

PP (32K prompt, eager kernels): cutlass f16 256x128 dense GEMMs 19.0%, turbo4 prompt_attn<5,1>
12.1% (now only the #2 kernel), mul_mat_q IQ3_S J=128 11.8% (3477 launches) + IQ4_NL J=128 6.8%
(3248) + J=112 tails 2.3%, gr_write_norm_rs 6.0%, gdn_rec_kh 5.9% (144 x 4.4ms), copy16_group 3.9%
(15,754 launches), gr_mix_r 3.3%, quantize_mmq_q8_1 2.5%, moe_combine4 2.0%. The GDN trio sums to
~15% (36/48 layers are GDN).

TG (320-token generation, --cuda-graph-trace=node; plain nsys shows NOTHING for decode - it runs in
CUDA graphs): **wait_flag_ge spin-waits 37.9%** (22,119 launches ~99/verify-round, avg 92.8us),
fetch_blobs 10.4% + copy_rows_from_mapped 1.0% (33 expert misses/round - this capture was
cache-cold, tg=54.9 vs 76.9% hit in house use), native_gu_multi IQ3_S 6.1%, small_mmvq Q8_0 4.1%
(drafter), native_down_multi IQ4_NL 2.8%, mmvq_multi ~6.9%, q6_k_mmvq 2.0% (350 x 302us),
attn_chunk<4,turbo4> 1.7%, gr_* ~5%. ~1558 launches per verify round. Reading: the decode pipeline
spends more GPU time waiting for expert arrivals than computing - the slot-count levers (kv-grow's
+550 slots, profile quality) are TG levers too, and the spin structure (99 handshakes/round) is the
biggest unexplored TG target.

MMQ J-cap analysis (buun e27405a44 concept): Strata's wrapper already sizes tiles per sorted
16-expert GROUP (ncols_opt = the group's max_rows, prefill.cpp:3360), not per ubatch - smarter than
buun's pre-fix code. At the 8192-token chunk / 512 experts / ~160 rows per expert, the J=128 waste
is the padded LAST tile (~37% of expert compute), and a narrow J would re-load weights per extra
column tile - buun only won because their sparse persistent kernels avoid that. Opt-in knob added:
STRATA_MMQ_JCAP=32|64 clamps ncols_opt in moe_mmq.cu (bitwise; J tiles the token dimension only).
VERDICT (jcap_probe, pp@31336, one boot per variant): off 2590.2, JCAP=64 2529.3 (-2.4%),
JCAP=32 2395.4 (-7.5%) - monotonically negative, exactly the weight-re-traffic failure mode: at
~160 rows/expert the narrow tiles re-load the expert weights more than they save in padding.
buun's +28% needed their sparse persistent kernels AND ~36 tokens/expert; neither holds here.
The knob stays in the tree default-off (it is 15 lines and documented), the J-cap idea is CLOSED
for this engine/model shape.

## kv-grow ADOPTED into the turbo4 config (10/16 evening, kv_grow_probe.json)

--kv-grow added to strata-huihui-iq3_s-turbo4.json (after the boundary fix 1a133a37 - without it
this config would fault at exactly 131072/262144 cells). One boot, matched against the static
B_896_auto variant of the config probe:

| case | static 896 | kv-grow | delta |
|---|---:|---:|---:|
| tg@46080 (256 tok) | 81.3 | **85.6** | **+5.3%** |
| tg@250000 (128 tok) | 82.7 | 82.1 | parity |
| pp@250K (matched size) | 2276.6 | 2272.4 | flat |
| expert slots at boot | 5474 | **6088** | +614 |
| draft layer at boot | 966 MiB | 834 MiB | drafter pool elastic too |
| #2065 warnings | 0 | 0 | - |

The mechanism, from the log: boot maps 23828 cells (0.15 of the 1.73 GiB pool); the 46K request
grows it to 55604 cells for 108 slots (36 hotter experts swapped into colder slots); the 250K
request grows to 246256 cells for 649 slots (232 swaps). At house depths the freed KV VRAM stays
in the expert cache - fewer misses, fewer fetches, fewer spins: that is the whole +5.3%. After a
deep request tg returns to ~82 (the pool stays grown, the cache rebalances) - no pathology.
The static-pool preference is reversed by measurement; --kv-grow stays in the config.

## Fuse knobs on CUDA: tested neutral (fuse_probe.jsonl + confirm_fuse, 10/16 late evening)

STRATA_FUSE_WAIT / STRATA_FUSE_HITS fold decode's spin launches into their consumer kernels
(wait+plan-copy into one launch; moe_hit_add into the combine; wait folded into the row copy's
blocks). Upstream defaults them ON for gfx12 only ("measured on gfx12"), OFF on NVIDIA. With
kv-grow in the config: 4-variant probe showed fuse_wait +1.2%, fuse_hits +0.8%, both +0.6% at
tg@46K - all within boot variance; the 6-boot alternating confirmation of the best (fuse_wait)
landed at +0.10% (base 83.75 vs fuse 83.83, n=6 each). Replies byte-identical everywhere (the
fusions do not move arithmetic). Verdict: NEUTRAL on this rig - under kv-grow the waits are
already short and the launch saving is not the wall. Not adopted; knobs left at defaults. The
6-boot base numbers (85.4-85.8 at 46K) also re-confirmed the kv-grow +5.3% across every boot.

## Backlog state after this session

1. ~~44K outlier~~ CLOSED (noise).
2. ~~PP decomposition~~ DONE (nsys; encode 67% -> 51% after the launch fix).
3. ~~Wide loads~~ DONE (committed, bitwise, -16% kernel).
4. ~~Raw-block staging~~ DONE for turbo4 (5b67fbda): 53 -> 28.5 KB, 3 blocks/SM, gather decode
   moved into the MMA fragment loads - mode 5 now beats q4_0's kernel (7.90 vs 9.02 ms/chunk) and
   the GPU-total penalty is gone. Extending RAW staging to turbo8/TCQ tiers is open (same pattern,
   bigger windows; TCQ needs the bit-unpack at fragment load).
5. ~~Encode cooperative rewrite~~ DONE (e2b45452): warp-per-group with the serial chains on lane 0
   - 1652.5 -> 17.5 ms, bitwise, PP penalty -8.4..-9.4% -> -4.1..-4.8%. The buun-flat alternative
   was not needed.
6. ~~Decode-side tables~~ DONE (1b61b630): qsa_decode_attn.cu's c_tq* codebooks moved from
   __constant__ to __device__ globals (divergent indices serialize constant-cache replays; the
   prompt gather escaped this in 94a8649). Value-identical, bitwise by construction; the parity
   harness's old path IS this kernel: turbo2_tcq decode reference 58.1 -> 38.0 ms/chunk (-34.5%),
   turbo4 flat (a 16-entry table was already cheap in the constant cache). Receipts: parity 12/12
   + kv_turbo_parity bitwise + ctest 20/20 + turbo4 smoke + needle 6/6 (incl. 3 at 125K on the
   final binary). The residual TG band (-1.7..-4.6%, matched acceptances) is decode-attention-
   adjacent for turbo4 no more - its fused decode reference costs the same as q4_0's (34.8 vs
   34.6 ms); treat TG as parity within draft variance until a 3-round run says otherwise.
7. NEW - rotation fusion (the +97 ms rotate slice): turbo_rotate128 is bandwidth-bound at DRAM
   peak (~1 TB/s effective on the Q rows) - kernel shape cannot help; only fusing the forward
   rotation into prompt_attn's q staging and the inverse into its epilogue removes the global
   round trips. Bitwise path exists (all three FWHT forms share the proven stage order), but the
   staging restructure is real surgery - candidate, not committed work.
8. ~~Drafter turbo KV~~ DONE (d1dd2aaf): the INT8 fallback is retired; draft layer 966 MiB, +62
   expert slots vs q4_0. See the section above; the TG/acceptance receipt is the A/B below it.
9. Gather LUT micro-variants: BOTH MEASURED NEGATIVE (10/16, reverted; bitwise in both cases -
   NEWHASHes unchanged). (a) smem table copy (0.5-2 KB static, LDS instead of divergent L1 LDG):
   turbo4 12.70 -> 12.55 (-1.7%, marginal), turbo8 15.27 -> 16.65 (+9% - the extra LDS contends
   with the score loop's own smem traffic), TCQ +1..+4%. (b) turbo4-only 16-entry register table
   via 2x __shfl_sync per code byte (uniform-call restructure, masked cells dequant row 0 with
   scale 0): turbo4 12.70 -> 13.18 (+3.5%), ctx1500 3.42 -> 3.63 (+6%). The issue-slot trade
   loses: post-wide-load, the L1-hot LUT reads are NOT the gather's wall. What remains of the
   +3.7 ms/chunk gap to q4_0 is the FP16 staging volume itself (256 B/item STS + the 53 KB
   footprint's 1-block/SM occupancy) = item 4's raw-block staging, as the 10/10 analysis said.
   Do not retry LUT relocation without new evidence.
10. NEW - DONE + ADOPTED: the elastic-KV overread tail (layer.cpp): pool_chunks and the
    reserve layout now carry +16 B, closing the exact-chunk-boundary fault (--kv-grow + turbo4 at
    131072 cells, and fp16/int8 every 256 slots upstream). Receipt: STRATA_KV_GROW=1 boot, prompt
    calibrated to prompt_n=131066 (the (131056,131072] window where slots*per_slot = 33 chunks
    exactly): no fault, coherent reply, pp 2490. --kv-grow then WON its benchmark (+5.3% tg@46K,
    +614 slots, parity at depth) and is now IN the turbo4 config - see the adoption section above.
    The live q4_0 config keeps the static pool (untouched; kv-grow there would need its own run).
11. NEW - MMQ J-cap: CLOSED, measured negative (see the decomposition section). STRATA_MMQ_JCAP
    knob in moe_mmq.cu, default off.
12. ~~Decode spin-wait structure~~ RE-DIAGNOSED (warm nsys, kv-grow config): the cold capture's
    37.9% was expert-arrival waiting - warm, wait_flag_ge is 3.2% (avg 92.8 -> 7.6us: the flags
    are already up). All-resident windows run the zero-doorbell graph (verify.cpp E-6, default
    since 0.1.39); spins only remain for miss windows. Warm decode is launch-bound instead:
    ~136 waits + ~200 small kernels (7-30us) per round - native_gu/down_multi 143/round,
    q6_k_mmvq 51/round (28.4us), block_topk_wide 13/round (56us), turbo4 attn_chunk 13/round
    (84.8us at 44K). Further TG work = fewer/fatter launches (the STRATA_LFUSE family is
    multi-GPU-only - needs ar_on(), dead end here) or fewer misses: STRATA_ROUTE_RESIDENT
    (residency-biased routing, margin in logits, default off) is the untested hit-rate lever -
    it changes routing, so it needs a quality gate, not just speed.
13. NEW - the GDN prefill trio ~15% of PP (gdn_rec_kh 5.9% at 4.4ms/instance, gr_write_norm_rs
    6.0%, gr_mix_r 3.3%): compare against buun's SM86 4-column GDN geometry (e27405a44 second
    half); Strata's kernels are its own, so this is a study, not a port.
14. NEW - copy16_group_kernel: 15,754 launches / 3.9% of PP at ~2.3 launches per MMQ call -
    expert-weight staging that may batch into fewer launches.
- The 10/09 ab.jsonl and the 10/10 ab_post_kernel.jsonl stand as history; the q4_0 baseline moved
  with the engine (TG +10-15% upstream), so cross-session comparisons use each session's own q4_0.

## TG under real content: prose vs code (10/16, HTTP, live serve config)

Probe, not an A/B: same running serve (turbo4, vram-reserve 896, spec 4, min-p 0.70), temp 0,
512-token completions, 4 rounds interleaved, per-round cache-busted prompts (cache_prompt false).
Harness: ../../tg_content_bench.py; raw: tg_content.jsonl. Shallow ctx (~50-token prompts) -
compare to tg@46K (81.3) in config_probe, NOT to each other across depths.

  case             tg t/s   draft acc
  prose_creative    97.8    0.69
  prose_expository 102.3    0.69
  prose_summarize  103.8    0.69
  code_python      111.2    0.76
  code_cpp         126.9    0.82
  code_web         133.0    0.90
  PROSE median     102.3    0.690
  CODE median      126.9    0.817

Read: code TG runs +24% over prose at shallow ctx, and the whole gap tracks draft acceptance
(0.82 vs 0.69) - highly templated content (HTML/CSS, C++ boilerplate) feeds the MTP head
predictable drafts; free prose does not. Spread is tight across rounds (+/-4%), seeds fixed.
Implication for spec tuning: min-p 0.70 was validated on synthetic dumps; prose-heavy traffic
sits at 0.69 acceptance, so any future min-p / spec-n retune should bench BOTH content classes.
