# TurboQuant KV port rebased onto upstream v0.1.45 (10/17)

State: git-side COMPLETE, receipts PENDING (build blocked on the running house serve).

## What happened

- Upstream moved 296b59d2 -> 477ac13c (264 commits, the f-145 perf-PR batch); v0.1.45 (annotated
  tag b39442e6) points at the tip.
- The old port/turbo-kv history was NOT linear: it contained the 10/16 merge commit 35c88047
  ("Merge origin/main (v0.1.44.1, 622 commits)"). `git rebase --onto upstream/main 296b59d2`
  flattened it: 38 non-merge commits replayed, merge-resolution-only content needed manual care.
- Backup: `backup/turbo-kv-prebase-20261017` (old tip b4655f65).
- New tip: e79c77b9 (39 commits: 38 replayed + 1 fixup). Force-push to origin pending receipts.
- rerere enabled this session (was off for the 10/16 merge).

## Conflicts resolved (commit 0c4e2afa -> 183d336f, engine wiring)

- verify.cpp: upstream folded the q/gate-split + norm_rope + rotation block into the `q_chain(cs)`
  lambda; the port's 2 remaining Q pre-rotation sites re-applied INSIDE q_chain on the lambda's
  `ns` stream (sites 1/append/inverse-rotate auto-merged). All 6 insertion points of the port
  commit re-audited present.
- qsa_decode_attn.cu: upstream refactored attn_chunk_kernel into `attn_chunk_body<KV_MODE,
  LANE_CELL, QUERY_SWIZZLE, REDUCE12>` (__device__) + three __global__ wrappers (kernel/slots/
  rows). Merged shape: TIER added LAST everywhere (`<..., REDUCE12, int TIER = 0>`); port's
  kv_mode-4 detection moved into upstream's `attn_kv_mode()` helper; the tier switch lives in
  `launch_chunks_batch` (early return), using upstream's `pre` (harness-names-its-kernel) instead
  of raw pre75_attn().
- NEW upstream opt-in paths gated for turbo (they carry no TIER; silent garbage otherwise):
  - qsa_decode_attn_slots (STRATA_QSA_SLOTS): `if (kv_mode == 4) return false;` - the caller in
    verify.cpp already falls back to per-query qsa_decode_attn_batch.
  - qsa_decode_attn_batch_rows (STRATA_BATCH_ATTN_ROWS): `per_row = per_row || kv_mode == 4;` -
    rides the existing per-row fallback through qsa_decode_attn_batch's tier dispatch.
- qsa_prompt_attn.cu (commit 0cda754b -> b51ccfcc): upstream added PF (#1821 prefetch) as the
  second template param of Smem/prompt_attn_kernel/launch; port's TIER moved LAST:
  `<KV_MODE, bool PF = false, int TIER = 0>`. Upstream's new IMMA v3 machinery kept whole.

## The trap that fired (pitfall #1, caught in audit)

- The port's auto-merged mode-5 dispatch read `launch<5, kTurbo4>` - under the merged signature
  that binds PF=kTurbo4 (int->bool, silently true) and TIER=0. Fixed to `launch<5, false, kTurbo4>`
  at all five tier sites.
- Same class in the RAW4 commit (5b67fbda -> 3319ff3a): `Smem<KV_MODE, TIER>` would bind PF=TIER;
  every Smem/kernel instantiation in the prompt_attn_kernel region now carries `<KV_MODE, PF,
  TIER>`; the port's raw-staging members renamed onto upstream's union member (S.k -> S.kv.k,
  S.v -> S.kv.v). Smem70/Smem3/SmemW (upstream's own structs with direct k/v members) untouched.
- Positional audit run over EVERY instantiation in both .cu files post-resolution.

## Evil-merge content lost + recovered

The 10/16 merge commit carried resolution-only content that no replayed commit contained:
- the TIER-LAST warning comment in qsa_decode_attn.cu -> restored in fixup e79c77b9.
That was the ONLY loss found by the audits below.

## Audits (pitfall #2)

- +line audit: 4640 code lines the port added vs old base; 30 not byte-present in the new delta -
  29 are the intended adaptations listed above (ns streams, PF/TIER signatures, launch<5,false,T>,
  early-return chain shape, `&& ok_im`), 1 was the lost comment (restored).
- -line audit: 81 port deletions; 15 re-appear in HEAD - all verified benign in context (identical
  lines living legitimately elsewhere: REDUCE12's own load8<KV_MODE> copy, Smem70's own KElem,
  the q4_0 else-arms of the port's own append chains, expression tails the backup also keeps).
- Port-only files (44 of 56): `git diff backup HEAD` EMPTY - byte-identical.
- Overlap files (12): deltas consistent with upstream's v0.1.44.1->v0.1.45 changes + adaptations.
- qsa_decode_attn.hpp: pure additions (upstream's Rows/SlotPools structs; QsaAttnPools layout
  untouched - no positional-init risk).

## Receipts (10/17, complete - house serve stopped for the duration, GPU exclusive)

- [x] build: cmake --build build -j32 - exit 0, [100%], binaries fresh (ccache made it ~1 min)
- [x] kv_turbo_parity: 6/6 ALL PASS (stashed 10/16 baseline binary AND fresh build)
- [x] qsa_prompt_attn_parity: 12/12 + the 3 new upstream IMMA arms PASS, FAILURES: 0;
      **NEWHASHes 12/12 BITWISE IDENTICAL** vs the stashed 10/16 binaries (default-path math
      unchanged by the merge; upstream's prompt-attn additions are opt-in as claimed)
- [x] ctest -R 'kv|qsa|turbo|snapshot|verify|prefill|mtp|gdn|stream' -j4: **28/28 passed** (62 s),
      incl. upstream's new qsa_slots_parity(_lanecell) + qsa_merge_daily_parity_{default,daily}
- [x] A/B, 3 rounds, arms q4_0,turbo4, FRESH control (ab.jsonl = r1, ab-rounds2-3.jsonl = r2-3):
  - PP (quote >=32K): pp@32K +1.0/+0.6/+0.9%; tg@250000-internal pp +1.3/+1.2/+1.3% - PARITY
    (10/16: +0.5/+0.4%). tg@46080-internal pp -8.8/+2.1/+5.0% - r1 outlier, median +2.1%,
    inside the documented ±15% machine-state wobble at 46K. tg@4096 pp = file-tier confound, N/A.
  - TTFT: parity at every depth (32K 7.4 vs 7.4; 250K 104.8 vs 103.5; 46K wobble both arms).
  - Quality gate: needle 18/18 turbo4 (3/3 at 32k + 128k, all 3 rounds), q4_0 18/18.
  - TG (band, 3 rounds): tg@4096 +23.3/+19.1/+18.5%; tg@46080 -8.8/-9.1/-12.7%;
    tg@250000 -7.9/-7.1/-4.7%.
  - **TG-at-depth is acceptance-driven, not engine speed**: q4_0 draft acceptance is BIT-STABLE
    vs 10/16 (17/38, 19/24, 26/41 - identical counts), while turbo4's moved: 46K 22/24 (91.7%)
    -> 20/27 (74.1%), 250K 24/37 -> 23/38, 4096 12/17 -> 21/28 (up). Spec decode preserves
    outputs (verify rejects bad drafts; needles all FOUND), so this is an efficiency flag, not
    a correctness flag. The 10/16 TG "parity band" itself swung -10.1%..+6.5% at these cases.
  - Seam audit for the acceptance move: mtp.cpp/prefill.cpp deltas re-checked line-by-line vs
    the port's original - all rotation/append seams intact, no duplicate split_attn call; the
    trajectory shift is upstream-engine-side (mtp batch-groups restructure / MoE scatter
    c675f5b7 "same bytes" claim / decode-path defaults), to decompose if it matters.
- [x] KL panel: SKIPPED per rule (port codec files byte-identical vs backup; no encode-path change)
- Verdict: merge VERIFIED for correctness and PP/TTFT parity; TG-at-depth acceptance shift
  flagged for backlog (run --spec 0 A/B or acceptance decomposition to isolate drafter vs main).

## Post-receipts

- Force-pushed origin/port/turbo-kv (rewritten history; backups: prebase-20261016/-20261017).
- House serve (huihui turbo4, port 9000) restarted on the NEW build after the receipts.
  NOTE: v0.1.45 serve counts thinking tokens against max_tokens (answer_room:false opts out) -
  watch reply lengths on clients that send small caps.

## Notes for the next session

- serve/server.py behavior change upstream: max_tokens now COUNTS thinking tokens (answer_room
  config opts out) - watch for truncated replies in A/B runs vs the 10/16 records.
- STRATA_QSA_SLOTS / STRATA_BATCH_ATTN_ROWS under turbo now fall back (per-query / per-row) -
  correct but unaccelerated; threading TIER through the slots/rows kernels is open backlog.
- launch_chunks_batch's turbo branch honors the `kernel` param (parity harness) via `pre`.
