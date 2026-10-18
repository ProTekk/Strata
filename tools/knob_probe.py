#!/usr/bin/env python3
"""Generic house-config knob probe: alternating boots of base vs one or more env arms on the LIVE
house serve config, tg@46080 x2 per boot (warm), reply-text + draft-counter capture (byte-identity
check for bitwise knobs), expert-cache hit rate and knob log lines scraped from the serve log.

    ~/Strata/.venv/bin/python tools/knob_probe.py --rounds 3 \
        --arm daily STRATA_ATTN_DAILY=1 --arm ram4 STRATA_RAM_ADAPT=4 \
        --out bench/results/<dir>/knob_probe.jsonl

Arms alternate within each round (order flips per round) to spread machine-state drift, the
kv-grow/route-resident protocol. Graceful wrapper stop per boot (keeps the shutdown summaries).
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

CFG = "/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.json"
LOG = Path("/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.log")
STRATA = Path.home() / "Strata"
PY = STRATA / ".venv/bin/python"
PORT = 9000
PARA = ("The lighthouse keeper recorded the weather each hour: wind from the northwest, "
        "sea moderate, visibility good. ")
PROMPT = PARA * (46080 // 22 + 2) + "\n\nSummarize the pattern in these records in two sentences."


def post(payload, timeout=1800.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def drain(timeout=180):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        out = subprocess.run(["pgrep", "-f", "server[.]py.*9000|strata --serv[e]|strata-visio[n]"],
                             capture_output=True, text=True).stdout.strip()
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if not out and used.isdigit() and int(used) < 1000:
            return True
        time.sleep(3)
    return False


def one_boot(extra, tag, round_i):
    log_start = LOG.stat().st_size if LOG.exists() else 0
    env = dict(os.environ)
    env.update(extra)
    proc = subprocess.Popen([str(PY), str(STRATA / "serve/server.py"), "--engine", "strata",
                             "--config", CFG, "--port", str(PORT)],
                            cwd=str(STRATA), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    row = {"tag": tag, "round": round_i, "env": extra}
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 420:
            if proc.poll() is not None:
                raise RuntimeError(f"boot failed rc={proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(2)
        row["boot_s"] = round(time.monotonic() - t0, 1)
        post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        tgs, texts, drafts = [], [], []
        for i in range(2):
            r = post({"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256,
                      "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            t = r.get("timings", {})
            tgs.append(t.get("predicted_per_second"))
            drafts.append([t.get("draft_n"), t.get("draft_n_accepted")])
            texts.append((r.get("choices") or [{}])[0].get("message", {}).get("content"))
        row["tg"] = tgs
        row["drafts"] = drafts
        row["text_sha"] = [__import__("hashlib").sha256((x or "").encode()).hexdigest()[:16] for x in texts]
        row["text0"] = (texts[0] or "")[:300]
        print(f"{tag} r{round_i}: tg {tgs} drafts {drafts}", flush=True)
        return row
    finally:
        proc.terminate()
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        txt = LOG.read_text(errors="replace")[log_start:]
        hits = re.findall(r"expert cache ([\d.]+)% hit", txt)
        row["hit_pct"] = hits[-3:] if hits else []
        row["knob_lines"] = [l.strip()[:160] for l in txt.splitlines()
                             if re.search(r"ram.?adapt|daily|tail skip", l, re.I)][:6]
        drain()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arm", action="append", nargs=2, metavar=("TAG", "ENV"),
                    help="arm tag and KEY=VAL (repeatable ENV within one --arm not supported; use several arms)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [("base", {})] + [(t, dict(kv.split("=", 1) for kv in [e])) for t, e in (a.arm or [])]
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert drain(), "drain failed before first boot"
    with open(out, "a") as f:
        for r in range(1, a.rounds + 1):
            order = arms if r % 2 else list(reversed(arms))
            for tag, env in order:
                row = one_boot(env, tag, r)
                f.write(json.dumps(row) + "\n")
                f.flush()
    # summary
    rows = [json.loads(l) for l in open(out)]
    print("\n=== summary (this run) ===")
    for tag, _ in arms:
        mine = [x for x in rows if x["tag"] == tag][-a.rounds:]
        tgs = [t for x in mine for t in x["tg"] if t]
        if tgs:
            print(f"{tag:8s} tg mean {sum(tgs)/len(tgs):6.2f}  vals {[round(t,1) for t in tgs]}")
        shas = {s for x in mine for s in x.get("text_sha", [])}
        print(f"{tag:8s} text hashes: {sorted(shas)}  drafts: {[x.get('drafts') for x in mine]}")


if __name__ == "__main__":
    main()
