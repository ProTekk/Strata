#!/usr/bin/env python3
"""Regression probe: the exact ab_kv_turbo tg case against ANY engine build, spec on or off.

Captures the completion text (temp 0 -> the trajectory) and the drafter acceptance counters, so
two builds can be compared token-for-token: identical spec-off texts + differing spec-on
acceptance = the drafter side moved; differing spec-off texts = the main model moved.

    ~/Strata/.venv/bin/python tools/kv_regress_probe.py --engine ~/strata-prebase/build/strata \
        --kv turbo4 --port 8091 --out bench/results/<dir>/regress [--no-spec] [--depth 46080]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STRATA = Path.home() / "Strata"
PY = STRATA / ".venv/bin/python"
LIVE = STRATA / "strata-unsloth-ud-iq4_xs.json"

PARAGRAPH = ("The lighthouse keeper recorded the weather each hour: wind from the northwest, "
             "sea moderate, visibility good. ")
PARAS_PER_TOKEN = 22


def build_prompt(target_tokens: int) -> str:
    return PARAGRAPH * (target_tokens // PARAS_PER_TOKEN + 2) + "\n\nSummarize the pattern in these records in two sentences."


def make_cfg(engine: Path, kv: str, port: int, out: Path, spec: bool, env: dict | None = None,
             tag_suffix: str | None = None) -> Path:
    cfg = json.loads(LIVE.read_text())
    cfg["exe"] = str(engine)
    a = cfg["args"]
    a[a.index("--kv") + 1] = kv
    if not spec:
        for flag in ("--spec", "--spec-min-p", "--mtp"):
            while flag in a:
                i = a.index(flag)
                del a[i:i + 2]
    if env:
        cfg["env"] = dict(cfg.get("env") or {}, **env)
    cfg["reasoning_budget_tokens"] = 512
    tag = f"{kv}-{'spec' if spec else 'nospec'}" + (f"-{tag_suffix}" if tag_suffix else "")
    cfg["model_name"] = f"probe-{tag}"
    cfg["log"] = str(out / f"probe-server-{tag}.log")
    cfg["port"] = port
    p = out / f"cfg-probe-{tag}.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1200) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server did not come up")


def post(port: int, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_case(port: int, depth: int, gen: int) -> tuple[dict, str]:
    payload = {"messages": [{"role": "user", "content": build_prompt(depth)}],
               "max_tokens": gen, "temperature": 0, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    rec: dict = {}
    text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            if d.get("timings"):
                rec = d
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            text += delta.get("content") or ""
    return rec, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--kv", default="turbo4")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=46080)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL merged into the cfg's engine env (repeatable)")
    ap.add_argument("--tag", default=None, help="output file tag suffix (default: derived from spec/kv)")
    a = ap.parse_args()
    engine = Path(a.engine).expanduser().resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = not a.no_spec
    tag = f"{a.kv}-{'spec' if spec else 'nospec'}" + (f"-{a.tag}" if a.tag else "")
    env = dict(kv.split("=", 1) for kv in a.env) if a.env else None
    cfg = make_cfg(engine, a.kv, a.port, out, spec, env, a.tag)
    proc = subprocess.Popen([str(PY), str(STRATA / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cfg), "--port", str(a.port)],
                            cwd=str(STRATA), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_ready(a.port, proc)
        post(a.port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
                      "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        rec, text = stream_case(a.port, a.depth, a.gen)
        t = rec.get("timings", {})
        row = {"engine": str(engine), "kv": a.kv, "spec": spec, "depth": a.depth,
               "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "tg": t.get("predicted_per_second"), "pp": t.get("prompt_per_second"),
               "text": text}
        (out / f"probe-{tag}-{engine.parent.parent.name}.json").write_text(json.dumps(row, indent=1))
        print(json.dumps({k: v for k, v in row.items() if k != "text"}))
        print("TEXT:", text[:400])
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
