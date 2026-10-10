#!/usr/bin/env python3
"""Teacher-forced KV-quality panel: the int8 (K8V8) reference vs the quantized arms, on fixed texts and a fixed expert set.

Method (the docs/UNSLOTH_Q4.md "Quality" recipe, engine-direct): the serve engine runs each text as
one `GEN 1 <ids>` request with STRATA_LOGPOS=<file> and STRATA_LOGPOS_TOPK=20, and --short-read sized
so the whole prompt is read through the SERIAL verify windows (windows_ok -> read_windows ->
Verifier::window_logprobs): one row per position with the teacher-forced logP(target), the argmax id,
and the top-20 (id:logprob). --prompt-cache 0 keeps the request one-shot.

Why not --dump-logits (the 10/09 failure): this is a native (IQ) pack. The engine refuses to start
without --prefill (the old base_args deleted it), and even with it a native pack never reaches the
per-token dump site (generate.cpp: "a native pack never reaches the per-token dump site" - the token
loop breaks straight into the verify windows), so the .bin dumps would be empty files.

Reference: `int8` - the highest supported Strata KV quality (K and V as INT8 codes + FP16 scale per 64,
layer.hpp's QsaState k_q/v_q/k_scale/v_scale; no rotation: STRATA_KV_ROT is unset, so kv_rot is false).
There is no `k8v8` string in --kv; `int8` IS K8V8. Every KL below is KL(int8 || arm), never KL(fp16 || arm).

Metrics per text and arm (all joined by position against the int8 dump):
  same top-1 %          the argmax id agrees with the int8 reference's
  mean KL(int8||arm)    over the reference's top-20 plus one bucket for the rest (the UNSLOTH_Q4.md
                        convention; an arm token outside its own top-20 carries the bucket's average
                        probability, which bounds the error at near-zero-mass terms)
  PPL                   exp(-mean logP(target)) - the target column needs no top-20 approximation
  dPPL nats/token       arm minus int8 (the ship criteria read the KL column)

    ~/Strata/.venv/bin/python tools/kv_kl_panel.py make      # build the .ids texts (tokenizer from ~/Strata/tools)
    ~/Strata/.venv/bin/python tools/kv_kl_panel.py run       # the dumps (engine direct; GPU must be free)
    ~/Strata/.venv/bin/python tools/kv_kl_panel.py compare   # the table (no GPU)

Texts: code (512), document (1024), chat (1024), document-8K (8192) - the 9/27 panel's set.
Expert determinism follows bench/results/2026-09-27-esp/esp_kl.py: --prefill KEPT (a native pack needs
it), --adapt-swaps 0, --adapt-every 0, --pcie-frac 0, --expert-cache 4000 (the arms must see the same
experts). Arms default to q4_0,turbo4,turbo2tcq against the int8 reference (override: --arms turbo4;
the reference always runs). Engine/config default to the port build and the live config. run()
validates every dump - exit code, timeout, row count, position sequence, target alignment - and
deletes a bad one, so `compare` can only see files that passed; compare refuses to run with any dump
missing. The texts, dumps (.logpos) and logs live under bench/results/2026-10-09-turbo-kv/kl/.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "bench" / "results" / "2026-10-09-turbo-kv" / "kl"
STRATA = Path.home() / "Strata"
sys.path.insert(0, str(STRATA / "tools"))          # strata_tokenizer (pure Python)

ENGINE = ROOT / "build" / "strata"
CONFIG = STRATA / "strata-unsloth-ud-iq4_xs.json"
TOKENIZER = Path.home() / "Strata-data/packs/unsloth-ud-iq4_xs/tokenizer"
TEXTS = {"code": 512, "doc": 1024, "chat": 1024, "doc8k": 8192}
REF = "int8"                                        # the K8V8 reference (see the docstring)
ARMS = ["q4_0", "turbo4", "turbo2tcq"]
TOPK = 20
COLS = 8                                            # the fixed STRATA_LOGPOS columns before the top-k pairs
N_VOCAB = 248320                                    # the pack tokenizer's vocab (vocab.json entries)


def tokenizer():
    from strata_tokenizer import Tokenizer
    vocab = json.loads((TOKENIZER / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TOKENIZER / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TOKENIZER / "token_type.json").read_text())
    return Tokenizer(tokens, merges, types)


def text_corpus(kind: str) -> str:
    if kind == "code":
        parts = []
        for p in sorted((ROOT / "src" / "kernels" / "cuda").glob("kv_*.cu")):
            parts.append(f"// === {p.name} ===\n" + p.read_text(errors="replace"))
        return "\n".join(parts)
    if kind in ("doc", "doc8k"):
        parts = []
        for p in sorted((ROOT / "docs").rglob("*.md")):
            parts.append(p.read_text(errors="replace"))
        if kind == "doc8k":     # the 8K arm needs more prose than docs/ alone holds
            for p in sorted((ROOT / "src").rglob("*.hpp")):
                parts.append(p.read_text(errors="replace"))
        return "\n\n".join(parts)
    if kind == "chat":
        turns = ["What changed in the KV cache this release?",
                 "The rotation is applied before quantization now - why does that help the 4-bit blocks?",
                 "Does it change the attention math?",
                 "How much VRAM does a 200K context cost with it on?",
                 "Walk me through what happens on the first token of a prompt.",
                 "And when the cache spills to RAM?",
                 "What would you measure before trusting it in production?",
                 "Fair. Anything that got weirder once it ran on the real model?"] * 12
        role = 0
        out = []
        for t in turns:
            out.append(("user: " if role == 0 else "assistant: ") + t)
            role ^= 1
        return "\n".join(out)
    raise SystemExit(f"unknown text {kind}")


def make():
    S.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    for kind, n in TEXTS.items():
        ids = tok.encode(text_corpus(kind), parse_special=True)[:n]
        if len(ids) < n:
            raise SystemExit(f"text {kind}: only {len(ids)} tokens available, need {n}")
        (S / f"{kind}.ids").write_text(",".join(str(i) for i in ids))
        print(f"{kind}: {len(ids)} tokens")


def base_args(cfg: dict) -> list[str]:
    """The live config's args, made deterministic. --prefill STAYS: a native pack refuses to start
    without it, and the serve path needs it for the prompt chunks between the verify windows."""
    a = list(cfg["args"])
    i = a.index("--expert-cache")
    a[i + 1] = "4000"                       # fixed slot count: the arms must see the same experts
    k = a.index("--kv")
    del a[k:k + 2]                           # the arm sets --kv
    return a + ["--pcie-frac", "0", "--adapt-swaps", "0", "--adapt-every", "0", "--prompt-cache", "0"]


def engine_command(args: list[str], arm: str, n_tokens: int) -> list[str]:
    """One arm's engine command. --short-read >= the prompt length is what routes the prompt through
    the serial verify windows (windows_ok), the only path that writes STRATA_LOGPOS rows."""
    if "--prefill" not in args:
        raise SystemExit("base_args lost --prefill: a native pack refuses to start without it")
    return [str(ENGINE), *args, "--kv", arm, "--serve", "--short-read", str(n_tokens + 64)]


def validate(path: Path, kind: str, n_tokens: int) -> str | None:
    """None when the dump is a complete, aligned window-logprob record of this text; else why not."""
    if not path.exists():
        return "no file"
    lines = [l for l in path.read_text().splitlines() if l]
    if len(lines) != n_tokens - 1:
        return (f"{len(lines)} rows, expected {n_tokens - 1} (a short count means the prompt did not "
                "run through the serial verify windows - check --short-read and the run log)")
    ids = [int(x) for x in (S / f"{kind}.ids").read_text().split(",")]
    for i, line in enumerate(lines):
        f = line.split("\t")
        if len(f) < COLS:
            return f"row {i}: {len(f)} columns, expected at least {COLS}"
        if len(f) == COLS:
            return f"row {i}: no top-k pairs - STRATA_LOGPOS_TOPK did not reach the engine"
        try:
            pos, tgt = int(f[0]), int(f[1])
        except ValueError:
            return f"row {i}: not tab-separated integers: {line[:60]!r}"
        if pos != i or tgt != ids[i + 1]:
            return f"row {i}: pos/target {pos}/{tgt} does not match the text ({i}/{ids[i + 1]})"
    return None


def run(arms: list[str], texts: list[str]):
    import os
    S.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    args = base_args(cfg)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", [])) + os.pathsep + env.get("PATH", "")
    todo = [REF] + [a for a in arms if a != REF]
    for kind in texts:
        n = TEXTS[kind]
        ids = (S / f"{kind}.ids").read_text()
        for arm in todo:
            out = S / f"{kind}-{arm}.logpos"
            if out.exists():
                why = validate(out, kind, n)
                if why is None:
                    continue
                print(f"{kind}/{arm}: existing dump rejected ({why}); rerunning", flush=True)
                out.unlink()
            # One engine per (text, arm): STRATA_LOGPOS opens its file once per process (static), and
            # one request per process keeps the position axis unambiguous. QUIT ends the serve loop
            # with exit 0; a request that fails prints "ERR ..." and keeps serving, so the log is
            # scanned for ERR too - a nonzero exit alone does not cover it.
            run_env = dict(env, STRATA_LOGPOS=str(out), STRATA_LOGPOS_TOPK=str(TOPK))
            cmd = engine_command(args, arm, n)
            t0 = time.monotonic()
            log_path = S / f"{kind}-{arm}.log"
            with open(log_path, "w") as log:
                try:
                    r = subprocess.run(cmd, input=f"GEN 1 {ids}\nQUIT\n".encode(),
                                       cwd=cfg.get("cwd", str(STRATA)),
                                       stdout=log, stderr=subprocess.STDOUT, env=run_env, timeout=7200)
                except subprocess.TimeoutExpired:
                    out.unlink(missing_ok=True)
                    raise SystemExit(f"{kind}/{arm}: engine timed out - see {log_path}")
            print(f"{kind} {arm}: exit {r.returncode} in {time.monotonic() - t0:.0f} s", flush=True)
            err_line = next((l for l in log_path.read_text().splitlines() if l.startswith("ERR ")), None)
            if err_line is not None:
                out.unlink(missing_ok=True)
                raise SystemExit(f"{kind}/{arm}: engine reported: {err_line} - see {log_path}")
            if r.returncode != 0:
                out.unlink(missing_ok=True)
                raise SystemExit(f"{kind}/{arm} failed (exit {r.returncode}) - see {log_path}")
            why = validate(out, kind, n)
            if why is not None:
                out.unlink(missing_ok=True)
                raise SystemExit(f"{kind}/{arm}: invalid dump ({why}) - see {log_path}")


def load(path: Path) -> dict[int, tuple[int, float, int, dict[int, float]]]:
    """pos -> (target, logP(target), argmax id, {id: logprob} of the top-k)."""
    rows: dict[int, tuple[int, float, int, dict[int, float]]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        f = line.split("\t")
        pos, tgt, logp, top = int(f[0]), int(f[1]), float(f[2]), int(f[3])
        topk = {}
        for pair in f[COLS:]:
            i, _, lp = pair.partition(":")
            topk[int(i)] = float(lp)
        rows[pos] = (tgt, logp, top, topk)
    return rows


def compare(arms: list[str], texts: list[str]):
    todo = [REF] + [a for a in arms if a != REF]
    missing = [f"{kind}/{arm}" for kind in texts for arm in todo if not (S / f"{kind}-{arm}.logpos").exists()]
    if missing:
        raise SystemExit("no comparison without every dump - run first, missing: " + ", ".join(missing))
    print(f"text     KV           top-1  KL(int8||arm)      PPL  dPPL nats/tok")
    for kind in texts:
        n = TEXTS[kind]
        ref_p = S / f"{kind}-{REF}.logpos"
        why = validate(ref_p, kind, n)
        if why is not None:
            raise SystemExit(f"{kind}/{REF}: reference dump invalid ({why})")
        ref = load(ref_p)
        nll_ref = -sum(r[1] for r in ref.values()) / len(ref)
        for arm in todo:
            if arm == REF:
                continue
            p = S / f"{kind}-{arm}.logpos"
            why = validate(p, kind, n)
            if why is not None:
                raise SystemExit(f"{kind}/{arm}: dump invalid ({why})")
            x = load(p)
            if set(x) != set(ref):
                raise SystemExit(f"{kind}/{arm}: position set differs from the reference")
            top1 = 0
            kl = 0.0
            nll = 0.0
            for pos, (tgt, logp_r, top_r, topk_r) in ref.items():
                _, logp_a, top_a, topk_a = x[pos]
                top1 += top_r == top_a
                nll -= logp_a
                # KL(ref || arm) over the reference's top-k plus one bucket for everything else.
                # A reference token outside the arm's top-k carries the arm bucket's average
                # probability (bucket mass / (vocab - k)); the vocab is ~150k, so that term is
                # bounded by p_ref * log((vocab - k) * other_a) and vanishes unless the arm puts
                # real mass on a token the reference ranks outside its top-20.
                s_r = sum(math.exp(lp) for lp in topk_r.values())
                s_a = sum(math.exp(lp) for lp in topk_a.values())
                b_r = max(0.0, 1.0 - s_r)
                b_a = max(1e-12, 1.0 - s_a)
                n_out = N_VOCAB - len(topk_a)         # the pack tokenizer's vocab minus the top-k
                q_floor = b_a / n_out
                for v, lp in topk_r.items():
                    p = math.exp(lp)
                    if p <= 0.0:
                        continue
                    q = math.exp(topk_a[v]) if v in topk_a else q_floor
                    kl += p * (lp - math.log(max(q, 1e-12)))
                if b_r > 0.0:
                    kl += b_r * math.log(b_r / b_a)
            n_rows = len(ref)
            ppl = math.exp(nll / n_rows)
            print(f"{kind:8s} {arm:10s} {100.0 * top1 / n_rows:5.1f}% {kl / n_rows:13.5f} {ppl:9.3f} "
                  f"{nll / n_rows - nll_ref:+14.3f}")
        print(f"{kind:8s} {REF + ' (ref)':10s} {'-':>7s} {'-':>13s} {math.exp(nll_ref):9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run", "compare"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--texts", default=",".join(TEXTS))
    a = ap.parse_args()
    arms = a.arms.split(",")
    texts = [t for t in a.texts.split(",") if t in TEXTS]
    if not texts:
        raise SystemExit(f"--texts: none of {a.texts} in {list(TEXTS)}")
    if a.cmd == "make":
        make()
    elif a.cmd == "run":
        run(arms, texts)
    else:
        compare(arms, texts)


if __name__ == "__main__":
    main()
