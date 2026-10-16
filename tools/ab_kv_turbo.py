#!/usr/bin/env python3
"""Phase 6 A/B: the KV arms through the server - tg at depth, pp, TTFT, and the needle pass.

Starts the serve wrapper once per arm per round (fresh engine, alternating order), then runs the plan's
matrix: tg @4K/46K/256K (long prompts built once, reused), pp @32K, TTFT at the client on the first
streamed delta (the community-benchmark convention), and the needle at 32K/128K via tools/needle_bench.py.
Throughputs come from the server's own timings block, never from wall-clock division.

    ~/Strata/.venv/bin/python tools/ab_kv_turbo.py [--rounds 3] [--arms q4_0,turbo4,turbo2tcq] [--port 8090]

Configs are generated per arm from the live config (exe -> the port build, --kv swapped, reasoning budget
set so answers are not truncated to thinking). Writes JSONL next to this file's --out (default
bench/results/2026-10-09-turbo-kv/ab.jsonl) and prints per-arm medians. The GPU must be free (the live
server stopped); each round costs roughly rounds x arms x (boot + ~15 min).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATA = Path.home() / "Strata"
S = ROOT / "bench" / "results" / "2026-10-09-turbo-kv"
ENGINE = ROOT / "build" / "strata"
LIVE = STRATA / "strata-unsloth-ud-iq4_xs.json"
PY = STRATA / ".venv" / "bin" / "python"

PARAGRAPH = ("The lighthouse keeper recorded the weather each hour: wind from the northwest, "
             "sea moderate, visibility good. ")
PARAS_PER_TOKEN = 22


def build_prompt(target_tokens: int) -> str:
    return PARAGRAPH * (target_tokens // PARAS_PER_TOKEN + 2) + "\n\nSummarize the pattern in these records in two sentences."


def post(port: int, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_ttft(port: int, payload: dict) -> tuple[dict, float | None]:
    """Stream one completion; return (final record with timings, client TTFT seconds)."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    ttft = None
    rec: dict = {}
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
            if ttft is None:
                ch = (d.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {})
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.monotonic() - started
    return rec, ttft


def arm_config(kv: str, port: int, s_dir: Path) -> Path:
    cfg = json.loads(LIVE.read_text())
    cfg["exe"] = str(ENGINE)
    a = cfg["args"]
    i = a.index("--kv")
    a[i + 1] = kv
    cfg["reasoning_budget_tokens"] = 512          # answers, not truncated thinking
    cfg["model_name"] = f"ab-{kv}"
    cfg["log"] = str(s_dir / f"server-{kv}.log")
    cfg["port"] = port
    p = s_dir / f"cfg-{kv}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
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


def one_arm_round(kv: str, port: int, out, rounds: int, round_i: int, s_dir: Path) -> None:
    cfg = arm_config(kv, port, s_dir)
    proc = subprocess.Popen([str(PY), str(STRATA / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cfg), "--port", str(port)],
                            cwd=str(STRATA), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, proc)
        post(port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})   # warm
        for depth, gen in ((4096, 128), (46080, 128), (250000, 96)):
            rec, ttft = stream_ttft(port, {"messages": [{"role": "user", "content": build_prompt(depth)}],
                                           "max_tokens": gen, "temperature": 0,
                                           "chat_template_kwargs": {"enable_thinking": False}})
            t = rec.get("timings", {})
            out.write(json.dumps({"round": round_i, "arm": kv, "case": f"tg@{depth}", "ttft_s": ttft,
                                  "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
                                  "tg": t.get("predicted_per_second"), "pp": t.get("prompt_per_second"),
                                  "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}) + "\n")
            out.flush()
        rec, ttft = stream_ttft(port, {"messages": [{"role": "user", "content": build_prompt(32768)}],
                                       "max_tokens": 1, "temperature": 0,
                                       "chat_template_kwargs": {"enable_thinking": False}})
        t = rec.get("timings", {})
        out.write(json.dumps({"round": round_i, "arm": kv, "case": "pp@32K", "ttft_s": ttft,
                              "prompt_n": t.get("prompt_n"), "pp": t.get("prompt_per_second")}) + "\n")
        out.flush()
        for length in ("32k", "128k"):
            r = subprocess.run([str(PY), str(ROOT / "tools" / "needle_bench.py"), "--lengths", length,
                                "--depths", "10,50,90", "--url", f"http://127.0.0.1:{port}"],
                               capture_output=True, text=True, timeout=7200)
            out.write(json.dumps({"round": round_i, "arm": kv, "case": f"needle@{length}",
                                  "needle": r.stdout.strip().splitlines()}) + "\n")
            out.flush()
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(10)          # let the GPU actually free before the next arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", default="q4_0,turbo4,turbo2tcq")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--dir", default=str(S), help="results dir for the generated cfgs/server logs")
    ap.add_argument("--out", default=None, help="JSONL path (default: <dir>/ab.jsonl)")
    a = ap.parse_args()
    arms = a.arms.split(",")
    s_dir = Path(a.dir).resolve()   # absolute: the log path enters the cfg, which the serve wrapper opens from its own cwd
    s_dir.mkdir(parents=True, exist_ok=True)
    out_path = a.out if a.out else str(s_dir / "ab.jsonl")
    with open(out_path, "a") as out:
        for rnd in range(1, a.rounds + 1):
            order = arms if rnd % 2 else list(reversed(arms))     # alternate to spread drift
            for kv in order:
                print(f"round {rnd} arm {kv}", flush=True)
                one_arm_round(kv, a.port, out, a.rounds, rnd, s_dir)
    # medians
    rows = [json.loads(l) for l in open(out_path)]
    print(f"\n{'case':10s} " + " ".join(f"{arm:>12s}" for arm in arms))
    for case in sorted({r["case"] for r in rows}, key=lambda c: (c.split("@")[0], c)):
        if case.startswith("needle"):      # needle records hold text lines, not a rate - summarized below
            continue
        line = f"{case:10s} "
        for arm in arms:
            vals = [(r.get("tg") or r.get("pp")) for r in rows
                    if r["arm"] == arm and r["case"] == case and (r.get("tg") or r.get("pp"))]
            line += f"{statistics.median(vals):12.1f}" if vals else f"{'-':>12s}"
        print(line)
    # TTFT medians (client-measured, seconds)
    print(f"\n{'case':10s} " + " ".join(f"{arm + ' TTFT':>12s}" for arm in arms))
    for case in sorted({r["case"] for r in rows if not r["case"].startswith("needle")}):
        line = f"{case:10s} "
        for arm in arms:
            vals = [r["ttft_s"] for r in rows if r["arm"] == arm and r["case"] == case and r.get("ttft_s")]
            line += f"{statistics.median(vals):12.2f}" if vals else f"{'-':>12s}"
        print(line)
    # needle: found-rate per arm per length (a needle line counts as found when it ends in OK/YES; see the raw
    # lines in the JSONL for the per-depth detail)
    print(f"\n{'needle':10s} " + " ".join(f"{arm:>12s}" for arm in arms))
    for case in sorted({r["case"] for r in rows if r["case"].startswith("needle")}):
        line = f"{case:10s} "
        for arm in arms:
            lines = [l for r in rows if r["arm"] == arm and r["case"] == case for l in r.get("needle", []) if "depth" in l]
            found = sum(1 for l in lines if ": FOUND" in l)
            line += f"{found:5d}/{len(lines):<6d}" if lines else f"{'-':>12s}"
        print(line)


if __name__ == "__main__":
    main()
