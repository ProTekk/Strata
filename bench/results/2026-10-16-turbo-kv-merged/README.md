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

## Backlog state after this session

1. ~~44K outlier~~ CLOSED (noise).
2. ~~PP decomposition~~ DONE (nsys; encode 67% -> 51% after the launch fix).
3. ~~Wide loads~~ DONE (committed, bitwise, -16% kernel).
4. Raw-block staging (53 -> ~28 KB, 3 blocks/SM): still open, now attacking a 38%-of-penalty
   slice; buun cp.async warning stands.
5. Encode cooperative rewrite: NOW THE TOP LEVER (51% of the remaining penalty, the spill is
   measured). Design choice open: warp-per-group with shared x[] (bitwise-preserving, the FWHT
   shuffle form is already proven equal) vs buun-flat threads-per-group with the spill (simplest,
   buun's shipped shape).
- The 10/09 ab.jsonl and the 10/10 ab_post_kernel.jsonl stand as history; the q4_0 baseline moved
  with the engine (TG +10-15% upstream), so cross-session comparisons use each session's own q4_0.
