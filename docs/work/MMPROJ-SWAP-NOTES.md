# f-mmproj-swap — vision on demand at 262k (branch notes, 2026-10-19)

Goal: the image encoder (strata-vision, measured 1,330 MiB GPU) pays rent only while a
picture is in flight, without shrinking `--max-context 262144`.

## What shipped (config, this branch)
- `vram_elastic: true` + `vision.elastic/elastic_mib 1536/elastic_idle_s 120` — the
  ca41a6b9 path: encoder starts with the first picture, parks after 120 s, cache keeps
  1536 MiB free for it via `POST /v1/vram`.
- **`--kv-grow` replaced by `--kv-resident 32768`** in both run configs. Measured cost of
  the elastic cache's assumption: with `--vram-elastic`, the engine turns `--kv-grow` off
  (generate.cpp:5039) and the whole 262k K/V pool becomes boot-resident — a ~1.5 GiB
  prepay (~800 expert slots) that ate the vision refund (measured: 7955 -> 7837 slots).
  `--kv-resident` is a different KV mode (K/V in pinned host RAM, ~1.9 GiB host, min
  20480 resident cells): it is not gated by `--vram-elastic` at all. 32768 keeps the
  whole 20220-token tetris window unstreamed. Context stays 262144, images stay elastic.

## v2 (the C++ work this branch should eventually carry)
Making `--kv-grow` coexist with `--vram-elastic`, which the engine forbids because the
segmented cache is not a single `VmmRange` (expert_cache.hpp:251-255: `#533` keeps its
own `reserved_/segs_` bookkeeping; `vmm_range()` is null -> `kvg_start` goes off at
generate.cpp:7586). The conflict is physical-granularity, not address space:
- kvg steals arena chunks at driver granularity (~2 MiB) and hands the handle to the K/V
  pool; segments are whole 512 MiB physicals. Handing a segment to the K/V (its grow step
  is ~58 MiB at 8192 cells) needs a K/V-side spare-chunk pool, and `/v1/vram` shrink must
  never release a physical whose pages live in the K/V range.
- Region analysis says they mostly don't collide: kvg works on slots below the prompt
  loan (`room below slot 5768`), while shrink takes arena-tail segments — and 1536 MiB of
  vision reserve fits inside the 4.12 GiB loan region, so the tail is free to take.
- Graph validity is preserved on both paths (addresses never move), so the invariant to
  protect is single ownership of each physical handle.

## Boot expectations to verify against
- `KV streaming: 131072 of 262144 cells in VRAM`-style line (resident x4) + pinned-RAM
  GiB; no `elastic K/V` line; no `--kv-grow is off` line.
- Expert cache slot count should land near 8500-8700 (the prepay gone, ~1.5 GiB back),
  with t4 printing ~80 slots fewer than q4_0 — turbo4's saving finally countable.
- tg for prompts/decode beyond ~32768 cells takes streamed host K/V: measure the tetris
  tail; pp/tg within 32k should be unchanged.
- First picture: 3-5 s stall, `strata-vision` in nvidia-smi ~1.3 GiB, gone 120 s later.
