#!/usr/bin/env python3
"""STRATA_ROUTE_RESIDENT probe: residency-biased routing margin (applied to rank lo-hi picks,
default 6-9). Alternating boots base vs margin 0.10 vs 0.25, tg@46080 x2 per boot; captures the
engine's expert-cache hit rate per boot from the serve log. Routing changes => replies may differ
(speed probe only; a quality gate is separate)."""
import json, os, re, subprocess, time, urllib.request
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

def one_boot(extra, tag):
    log_start = LOG.stat().st_size if LOG.exists() else 0
    env = dict(os.environ); env.update(extra)
    proc = subprocess.Popen([str(PY), str(STRATA / "serve/server.py"), "--engine", "strata",
                             "--config", CFG, "--port", str(PORT)],
                            cwd=str(STRATA), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
        post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        tgs = []
        for i in range(2):
            r = post({"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256,
                      "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            t = r.get("timings", {})
            tgs.append(t.get("predicted_per_second"))
        print(f"{tag}: tg {tgs}", flush=True)
        return tgs
    finally:
        proc.terminate()
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
        txt = LOG.read_text(errors="replace")[log_start:]
        hits = re.findall(r"expert cache ([\d.]+)% hit", txt)
        rr = [l.strip()[:130] for l in txt.splitlines() if "route" in l.lower() and "resident" in l.lower()][:3]
        print(f"{tag}: hit rates seen: {hits[-4:] if hits else 'none'}; rr lines: {rr}", flush=True)
        for _ in range(45):
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip()
            if used.isdigit() and int(used) < 1000:
                break
            time.sleep(2)

variants = [("base", {}), ("rr010", {"STRATA_ROUTE_RESIDENT": "0.10"}),
            ("rr025", {"STRATA_ROUTE_RESIDENT": "0.25"})]
res = {t: [] for t, _ in variants}
for rnd in range(2):
    for tag, extra in variants:
        res[tag] += one_boot(extra, f"round{rnd} {tag}")
print()
for tag, _ in variants:
    v = res[tag]
    print(f"{tag}: n={len(v)} mean={sum(v)/len(v):.2f} vals={v}")
