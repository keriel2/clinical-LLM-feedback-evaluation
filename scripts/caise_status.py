"""
caise_status.py: progress of every CAiSE run in one table.

For each run: state (waiting / training / evaluating / done / failed), progress as a percentage of
training steps or rounds, elapsed time, a rough time to finish, and the GPUs it was given.
It reads the queue log and each run's train.log / eval.log; it changes nothing.

Usage (from .):   python3 caise_status.py
Refresh every 10 minutes:               watch -n 600 python3 caise_status.py
"""
import os
import re
import glob
import time
from datetime import datetime

ROOT = "ablation_caise"
QUEUE = os.path.join(ROOT, "queue.log")
# "... step 1200 / 7500", "round 300/3750", "Progress: 340/1273"
FRAC = re.compile(r"(?:step|[Rr]ound|Progress:?)\s*(\d+)\s*/\s*(\d+)")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def last_frac(path):
    if not os.path.exists(path):
        return None
    hit = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = FRAC.search(line)
            if m:
                hit = (int(m.group(1)), int(m.group(2)))
    return hit


def tail_has(path, words, n=40):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as f:
        lines = f.readlines()[-n:]
    return any(w in l for l in lines for w in words)


def queue_info():
    launched, order = {}, []
    if os.path.exists(QUEUE):
        for line in open(QUEUE, errors="ignore"):
            m = re.search(r"\[(.+?)\] launched (\S+) on GPUs (\S+)", line)
            if m:
                launched[os.path.basename(m.group(2))] = (m.group(1), m.group(3))
    return launched


def fmt(sec):
    if sec is None or sec < 0:
        return "-"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m"


def main():
    launched = queue_info()
    runs = sorted(set(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "*_s*")) if os.path.isdir(d)) | set(launched))
    now = time.time()
    print(f"{'run':30s} {'state':11s} {'progress':>14s} {'elapsed':>8s} {'to finish':>9s}  gpus")
    for r in runs:
        d = os.path.join(ROOT, r)
        cond = r.rsplit("_", 2)[0] if r.count("_") >= 2 else r
        tlog, elog = os.path.join(d, "train.log"), os.path.join(d, "eval.log")
        done = bool(glob.glob(os.path.join(d, "*", "eval_results.json")))
        start, gpus = launched.get(r, (None, "-"))
        t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp() if start else None
        elapsed = now - t0 if t0 else None
        prog, eta = "", None
        if done:
            state = "done"
        elif tail_has(tlog, ["Traceback", "Error", "out of memory"]) or tail_has(elog, ["Traceback", "Error"]):
            state = "FAILED"
        elif os.path.exists(elog) and os.path.getsize(elog) > 0:
            state = "evaluating"
            f = last_frac(elog)
            prog = f"{f[0]}/{f[1]}" if f else ""
        elif os.path.exists(tlog):
            state = "training"
            f = last_frac(tlog)
            if f and f[1]:
                pct = f[0] / f[1]
                prog = f"{100 * pct:5.1f}% ({f[0]})"
                if elapsed and pct > 0.01:
                    eta = elapsed / pct - elapsed + 2.5 * 3600   # + ~2.5 h evaluation
        else:
            state = "waiting"
        print(f"{r:30s} {state:11s} {prog:>14s} {fmt(elapsed):>8s} {fmt(eta):>9s}  {gpus}")
    qd = tail_has(QUEUE, ["queue done"])
    print(f"\nqueue: {'finished' if qd else 'running'}   ({QUEUE})")
    os.system("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'")


if __name__ == "__main__":
    main()
