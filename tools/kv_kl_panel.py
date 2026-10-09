#!/usr/bin/env python3
"""Teacher-forced KV-quality panel: fp16 reference vs the quantized arms, on fixed texts and a fixed expert set.

The successor to tools/kv_precision_compare.py (the 9/27 q4_0 receipts tool, since removed): same method,
TurboQuant tiers added. For each arm the engine runs the same token ids with --max-new 1 and --dump-logits,
so every position gets a logits row from the same model, same weights, same experts - only --kv differs.
Against the fp16 dump the panel reports: same-top-1 %, mean KL(fp16 || arm), perplexity, and the perplexity
delta in nats/token (the ship criteria read the KL column).

    ~/Strata/.venv/bin/python tools/kv_kl_panel.py make      # build the .ids texts (tokenizer from ~/Strata/tools)
    ~/Strata/.venv/bin/python tools/kv_kl_panel.py run       # the dumps (engine direct; GPU must be free)
    ~/Strata/.venv/bin/python tools/kv_kl_panel.py compare   # the table (no GPU)

Texts: code (512), document (1024), chat (1024), document-8K (8192, logits stride 8) - the 9/27 panel's set.
Expert determinism follows bench/results/2026-09-27-esp/esp_kl.py: --prefill dropped (one token at a time,
a logits row per position), --adapt-swaps 0, --pcie-frac 0, expert-cache fixed. Arms default to
fp16,int8,q4_0,turbo4,turbo2tcq (override: --arms fp16,turbo4). Engine/config default to the port build and
the live config; the texts and dumps live under bench/results/2026-10-09-turbo-kv/kl/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = ROOT.parent / "strata-tq-port" / "bench" / "results" / "2026-10-09-turbo-kv" / "kl"
STRATA = Path.home() / "Strata"
sys.path.insert(0, str(STRATA / "tools"))          # strata_tokenizer (pure Python)

ENGINE = ROOT / "build" / "strata"
CONFIG = STRATA / "strata-unsloth-ud-iq4_xs.json"
TOKENIZER = Path.home() / "Strata-data/packs/unsloth-ud-iq4_xs/tokenizer"
TEXTS = {"code": 512, "doc": 1024, "chat": 1024, "doc8k": 8192}
STRIDES = {"code": 1, "doc": 1, "chat": 1, "doc8k": 8}
ARMS = ["fp16", "int8", "q4_0", "turbo4", "turbo2tcq"]


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
    a = list(cfg["args"])
    i = a.index("--expert-cache")
    a[i + 1] = "4000"                       # fixed slot count: the arms must see the same experts
    j = a.index("--prefill")                 # one token at a time: a logits row per position
    del a[j:j + 2]
    k = a.index("--kv")
    del a[k:k + 2]                           # the arm sets --kv
    return a + ["--pcie-frac", "0", "--adapt-swaps", "0", "--adapt-every", "0"]


def run(arms: list[str]):
    import os
    S.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    args = base_args(cfg)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", [])) + os.pathsep + env.get("PATH", "")
    for kind in TEXTS:
        for arm in arms:
            out = S / f"{kind}-{arm}.bin"
            if out.exists():
                continue
            cmd = [str(ENGINE), *args, "--kv", arm, "--tokens-file", str(S / f"{kind}.ids"),
                   "--max-new", "1", "--logits-stride", str(STRIDES[kind]), "--dump-logits", str(out)]
            with open(S / f"{kind}-{arm}.log", "w") as log:
                r = subprocess.run(cmd, cwd=cfg.get("cwd", str(STRATA)), stdout=log, stderr=subprocess.STDOUT, env=env)
            print(kind, arm, "exit", r.returncode, flush=True)
            if r.returncode != 0:
                raise SystemExit(f"{kind}/{arm} failed - see {S}/{kind}-{arm}.log")


def load(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n_vocab, n_rows = np.frombuffer(raw[:8], dtype=np.int32)
    a = np.frombuffer(raw[8:], dtype=np.float32)
    if a.size != int(n_vocab) * int(n_rows):
        raise SystemExit(f"{path}: header says {n_vocab}x{n_rows}, file holds {a.size} floats")
    return a.reshape(int(n_rows), int(n_vocab))


def logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(axis=-1, keepdims=True))


def compare(arms: list[str]):
    print(f"{'text':8s} {'KV':10s} {'top-1':>7s} {'mean KL':>9s} {'PPL':>9s} {'dPPL nats/tok':>14s}")
    for kind in TEXTS:
        ref_p = S / f"{kind}-fp16.bin"
        if not ref_p.exists():
            print(f"{kind}: no fp16 dump - run first")
            continue
        ref = load(ref_p)
        lp_ref = np.stack([logsoftmax(r) for r in np.split(ref, 16)])   # 16 chunks: the vocab fits in RAM once
        lp_ref = np.concatenate(lp_ref)
        t1_ref = lp_ref.argmax(-1)
        nll_ref = -np.take_along_axis(lp_ref, t1_ref[:, None], -1).ravel()
        for arm in arms:
            p = S / f"{kind}-{arm}.bin"
            if arm == "fp16" or not p.exists():
                continue
            x = load(p)
            if x.shape != ref.shape:
                print(f"{kind:8s} {arm:10s} SKIP: {x.shape} vs fp16 {ref.shape}")
                continue
            lp = np.stack([logsoftmax(r) for r in np.split(x, 16)])
            lp = np.concatenate(lp)
            t1 = lp.argmax(-1)
            same = float((t1 == t1_ref).mean())
            p_ref = np.exp(lp_ref)
            kl = float((p_ref * (lp_ref - lp)).sum(-1).mean())
            nll = float(-np.take_along_axis(lp, t1_ref[:, None], -1).ravel().mean())
            ppl = float(np.exp(nll))
            d = nll - float(nll_ref.mean())
            se = float(np.std(np.take_along_axis(lp, t1_ref[:, None], -1).ravel()
                              - np.take_along_axis(lp_ref, t1_ref[:, None], -1).ravel()) / np.sqrt(len(t1_ref)))
            print(f"{kind:8s} {arm:10s} {100 * same:6.1f}% {kl:9.4f} {ppl:9.3f} {d:+9.3f} +- {se:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run", "compare"])
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    arms = a.arms.split(",")
    if a.cmd == "make":
        make()
    elif a.cmd == "run":
        run(arms)
    else:
        compare(arms)


if __name__ == "__main__":
    main()
