"""Exact McNemar per benchmark for pairs of eval_results.json files, with Holm correction.
Usage: python paired_tests.py LABEL_A=path_a.json LABEL_B=path_b.json [LABEL_C=... LABEL_D=...]
Pairs are taken in order: (A,B), (C,D), ...  Benchmark key 'biasmedqa' is the five-option MedQA test set."""
import json, sys
from math import comb

def load(p):
    d = json.load(open(p))
    return {b: {e["id"]: bool(e["correct"]) for e in v["per_example"]} for b, v in d.items()}

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

args = [a.split("=", 1) for a in sys.argv[1:]]
rows = []
for (la, pa), (lb, pb) in zip(args[0::2], args[1::2]):
    A, B = load(pa), load(pb)
    for bench in ("medqa", "biasmedqa", "pubmedqa"):
        ids = A[bench].keys() & B[bench].keys()
        b = sum(A[bench][i] and not B[bench][i] for i in ids)
        c = sum(B[bench][i] and not A[bench][i] for i in ids)
        accA = 100 * sum(A[bench][i] for i in ids) / len(ids)
        accB = 100 * sum(B[bench][i] for i in ids) / len(ids)
        rows.append([f"{la} vs {lb}", {"biasmedqa": "medqa5", "medqa": "medqa4"}.get(bench, bench),
                     accA, accB, b, c, mcnemar(b, c)])
order = sorted(range(len(rows)), key=lambda i: rows[i][6])
m, running = len(rows), 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * rows[i][6]))
    rows[i].append(running)
print(f"{'contrast':38s} {'bench':8s} {'accA':>6s} {'accB':>6s} {'A>B':>5s} {'B>A':>5s} {'p':>7s} {'p_holm':>7s}")
for r in rows:
    print(f"{r[0]:38s} {r[1]:8s} {r[2]:6.2f} {r[3]:6.2f} {r[4]:5d} {r[5]:5d} {r[6]:7.4f} {r[7]:7.4f}")
