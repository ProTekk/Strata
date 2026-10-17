#!/usr/bin/env python3
"""RR=0.25 quality gate: needle 6/6 (32k+128k), tg@250K, prose sanity, hit-rate + swap stats."""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

CFG = "/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.json"
LOG = Path("/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.log")
STRATA = Path.home() / "Strata"
PY = STRATA / ".venv/bin/python"
PORT = 9000
PARA = ("The lighthouse keeper recorded the weather each hour: wind from the northwest, "
        "sea moderate, visibility good. ")

def post(payload, timeout=1800.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# drain any stragglers first
for _ in range(60):
    out = subprocess.run(["pgrep", "-f", "server[.]py.*9000|strata --serv[e]"],
                         capture_output=True, text=True).stdout.strip()
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    if not out and used.isdigit() and int(used) < 1000:
        break
    time.sleep(3)
print(f"drained; gpu={used} MiB", flush=True)

log_start = LOG.stat().st_size if LOG.exists() else 0
env = dict(os.environ); env["STRATA_ROUTE_RESIDENT"] = "0.25"
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
    print(f"ready at {time.monotonic()-t0:.0f}s", flush=True)

    needle = subprocess.run([str(PY), "tools/needle_bench.py", "--lengths", "32k,128k",
                             "--depths", "10,50,90", "--url", f"http://127.0.0.1:{PORT}"],
                            cwd="/home/jacob/strata-tq-port", capture_output=True, text=True, timeout=2400)
    print("=== needle ===")
    print(needle.stdout[-900:], flush=True)

    r = post({"messages": [{"role": "user", "content": PARA * (250000 // 22 + 2) +
              "\n\nWhat is the pattern? One sentence."}], "max_tokens": 128,
              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    t = r.get("timings", {})
    print(f"tg@250K: prompt_n={t.get('prompt_n')} pp={t.get('prompt_per_second')} "
          f"tg={t.get('predicted_per_second')} draft={t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
    print("reply:", (r["choices"][0]["message"].get("content") or "").strip()[:120], flush=True)

    r = post({"messages": [{"role": "user", "content":
              "Write a four-sentence analysis of why lighthouse keepers kept meticulous weather "
              "records, and what historians can infer from them."}], "max_tokens": 256,
              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    prose = (r["choices"][0]["message"].get("content") or "").strip()
    print("prose sanity:", prose[:300], flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()

txt = LOG.read_text(errors="replace")[log_start:]
hits = re.findall(r"hit rate: ([\d.]+)%", txt)
rr = [l.strip()[:160] for l in txt.splitlines() if l.startswith("route-resident")][:4]
warns = sum(1 for l in txt.splitlines() if "WARNING" in l)
print(f"hit rates: {hits[-6:]}")
print(f"warnings: {warns}")
for l in rr:
    print(" ", l)
print("RR25 GATE DONE")
