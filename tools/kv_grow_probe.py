#!/usr/bin/env python3
"""kv-grow probe: one boot of the turbo4 config (now with --kv-grow), the same cases as the
static B_896_auto variant for direct comparison, plus a repeat short-tg after the deep request
to expose churn/regrow. Captures boot lines, warnings, and the elastic grow log lines."""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

CFG = "/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.json"
LOG = "/home/jacob/strata-tq-port/strata-huihui-iq3_s-turbo4.log"
STRATA = Path.home() / "Strata"
PY = STRATA / ".venv/bin/python"
PORT = 9000
PARA = ("The lighthouse keeper recorded the weather each hour: wind from the northwest, "
        "sea moderate, visibility good. ")

def build_prompt(target):
    return PARA * (target // 22 + 2) + "\n\nSummarize the pattern in these records in two sentences."

def post(payload, timeout=1800.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# mark the log position so we only read THIS boot's lines
log_start = Path(LOG).stat().st_size if Path(LOG).exists() else 0

proc = subprocess.Popen([str(PY), str(STRATA / "serve/server.py"), "--engine", "strata",
                         "--config", CFG, "--port", str(PORT)],
                        cwd=str(STRATA), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
results = []
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
    print(f"boot: {time.monotonic()-t0:.0f}s", flush=True)
    post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})

    cases = [
        ("tg@46080_first", 46080, 256),
        ("tg@250000", 250000, 128),
        ("tg@46080_after_deep", 46080, 256),
        ("pp@32K", 32768, 1),
    ]
    for tag, depth, gen in cases:
        r = post({"messages": [{"role": "user", "content": build_prompt(depth)}], "max_tokens": gen,
                  "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        t = r.get("timings", {})
        rec = {"case": tag, "prompt_n": t.get("prompt_n"), "pp": t.get("prompt_per_second"),
               "tg": t.get("predicted_per_second"), "draft": f"{t.get('draft_n_accepted')}/{t.get('draft_n')}"}
        results.append(rec)
        print(json.dumps(rec), flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()

txt = Path(LOG).read_text(errors="replace")[log_start:]
boot_lines = [l.split("generate: ")[-1] for l in txt.splitlines()
              if re.search(r"expert cache auto|elastic K/V|draft layer", l)]
warns = sum(1 for l in txt.splitlines() if "WARNING" in l)
grow_events = [l.strip()[:150] for l in txt.splitlines() if re.search(r"kvg|grow|gives the K/V|K/V takes", l, re.I)]
print("\n=== boot lines ===")
for l in boot_lines[:6]:
    print(" ", l[:160])
print(f"warnings: {warns}")
print("grow-related lines (first 10):")
for l in grow_events[:10]:
    print(" ", l)
Path("/home/jacob/strata-tq-port/bench/results/2026-10-16-turbo-kv-merged/kv_grow_probe.json").write_text(
    json.dumps({"results": results, "boot_lines": boot_lines, "warnings": warns,
                "grow_events": grow_events[:40]}, indent=1))
print("KV GROW PROBE COMPLETE")
