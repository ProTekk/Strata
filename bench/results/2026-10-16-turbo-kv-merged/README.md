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
8. Gather LUT micro-variants: BOTH MEASURED NEGATIVE (10/16, reverted; bitwise in both cases -
   NEWHASHes unchanged). (a) smem table copy (0.5-2 KB static, LDS instead of divergent L1 LDG):
   turbo4 12.70 -> 12.55 (-1.7%, marginal), turbo8 15.27 -> 16.65 (+9% - the extra LDS contends
   with the score loop's own smem traffic), TCQ +1..+4%. (b) turbo4-only 16-entry register table
   via 2x __shfl_sync per code byte (uniform-call restructure, masked cells dequant row 0 with
   scale 0): turbo4 12.70 -> 13.18 (+3.5%), ctx1500 3.42 -> 3.63 (+6%). The issue-slot trade
   loses: post-wide-load, the L1-hot LUT reads are NOT the gather's wall. What remains of the
   +3.7 ms/chunk gap to q4_0 is the FP16 staging volume itself (256 B/item STS + the 53 KB
   footprint's 1-block/SM occupancy) = item 4's raw-block staging, as the 10/10 analysis said.
   Do not retry LUT relocation without new evidence.
- The 10/09 ab.jsonl and the 10/10 ab_post_kernel.jsonl stand as history; the q4_0 baseline moved
  with the engine (TG +10-15% upstream), so cross-session comparisons use each session's own q4_0.
