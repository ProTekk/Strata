# v0.1.45 knob probes for the house config (10/17 evening)

Protocol: tools/knob_probe.py - alternating boots of the LIVE house config (huihui turbo4,
route-resident 0.25, kv-grow, spec 4/min-p 0.70) base vs arm, 3 rounds, tg@46080 x2 per boot
(warm, max_tokens 256, temp 0, thinking off), reply-text sha + draft counters captured per run,
expert-cache hit % scraped from the serve log. GPU exclusive, graceful wrapper stops.

## STRATA_ATTN_DAILY=1 - ADOPTED (+3.5%, bit-exact)

PR #540's two-cells-per-step chunk kernel (attn_chunk_kernel_pre75) for decode attention -
verify windows AND the MTP drafter - plus attn_merge_v2 (skips chunks marked empty).

| arm  | tg values (6 boots, alternating)          | mean  |
|------|-------------------------------------------|-------|
| base | 85.4 88.5 / 85.3 88.2 / 85.2 88.2         | 86.80 |
| daily| 88.2 91.4 / 88.2 91.6 / 88.3 91.6         | 89.88 |

- **+3.5% mean, every pairwise comparison daily > base, distributions non-overlapping.**
- Bit-exact verified EMPIRICALLY: reply-text sha256 identical across arms (128fc8a2f4de6ef5 /
  3b093fc05109869a both arms) AND draft counters identical in all 12 runs ([29,22], [21,18]) -
  the drafter's attention rides the same kernel, so acceptance would move if a bit moved.
- Consistent with construction (upstream: "bit for bit with the default kernels by construction")
  and with qsa_merge_daily_parity_{default,daily} passing in the post-merge ctest (28/28).
- Turbo interaction: the port's tier dispatch honors `pre` (launch_chunks_batch) so mode 4 runs
  pre75<M,8,true,TIER> under the flag - the probes are the receipt.
- 250K depth: not separately gated here (parity harness covers chunk-border widths; adoption
  boot + a 250K sanity prompt recorded below).

## STRATA_RAM_ADAPT=4 - REJECTED (dead neutral)

Resident RAM copy follows decode routing (d7298883, port of #1838 + #2134 foresight fix).

| arm  | tg values                                  | mean  |
|------|--------------------------------------------|-------|
| base | 84.3 88.6 / 85.5 88.3 / 85.2 88.1          | 86.67 |
| ram4 | 84.8 88.3 / 85.2 88.1 / 85.3 88.3          | 86.67 |

- 86.67 vs 86.67 - zero delta, identical texts/drafts. Warm decode on this rig is launch-bound
  (~136 waits + ~200 small kernels/round), not expert-arrival-bound; the 37.9%-spin profile was
  CACHE-COLD. With pcie-frac 0.00 + the VRAM cache's hit rate, RAM-copy placement has nothing
  left to fix in the warm regime. Same class as the fuse knobs (+0.10% -> rejected 10/16).
- Do not re-test without a cold-cache or file-tier-heavy regime (e.g., a bigger pack or a
  deliberately cold boot profile).

## Adoption

- strata-huihui-iq3_s-turbo4.json: env gains "STRATA_ATTN_DAILY": "1" (alongside
  STRATA_ROUTE_RESIDENT 0.25). Revert = delete the key.
- Post-adoption boot: engine environ verified (/proc/<pid>/environ: ATTN_DAILY=1 +
  ROUTE_RESIDENT=0.25), model loaded, smoke reply clean.
- 250K sanity on the adopted config: prompt_n 238690, pp 2386.3 t/s, tg 88.0, drafts 25/28
  accepted, coherent reply, 100.6 s wall - the daily kernel + merge_v2 are clean at house max
  depth (262144).

## Not probed (screened out on paper)

ADAPT_NOWAIT/LAG (correctness knob for a mode the house doesn't run; upstream's own measurement
is -1.1%, noise), PCIE_BALANCE (inert at pcie-frac 0.00), IMMA/Q4_TC/SM75 (int8-KV/Turing),
HC_*/q36speed (Qwen3.6 family), MMQ_TAB/GROUP_SORT/B6_MMVQ_ROWS (PP-side; the JCAP result says
narrow-tile shapes are negative on this geometry), QSA_SLOTS/BATCH_ATTN_ROWS (gated to fallback
under turbo by the merge - no benefit until TIER threading), KV_HOST_REGISTER (WSL), SPLIT_*/P2P
(multi-GPU). FILE_CACHE_OLD_RULE: fallback only, if file-cache retention ever looks worse.
