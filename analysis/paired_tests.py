"""Exact two-sided McNemar tests between runs, with Holm adjustment within one family.

Library use:   from paired_tests import compare_family
Command line:  python paired_tests.py A=path_a.json B=path_b.json [C=... D=...]
               (pairs are (A,B), (C,D), ...; Holm is applied across all tests in the call)

Pairing is by question identifier. Before testing, the script checks that every paired
identifier has the same gold answer in both runs, so that a difference in question order
between evaluations would be detected rather than silently mis-paired.
Benchmark key 'biasmedqa' is the five-option MedQA test set (no added bias content)."""
import json, sys
from math import comb

BENCH = {"medqa": "medqa4", "biasmedqa": "medqa5", "pubmedqa": "pubmedqa"}

def load(path):
    d = json.load(open(path))
    return {b: {e["id"]: (bool(e["correct"]), e["gold"]) for e in v["per_example"]}
            for b, v in d.items() if isinstance(v, dict) and "per_example" in v}

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running, m = [0.0] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj

def compare(label_a, path_a, label_b, path_b):
    A, B = load(path_a), load(path_b)
    rows = []
    for bench, name in BENCH.items():
        ids = A[bench].keys() & B[bench].keys()
        mism = [i for i in ids if A[bench][i][1] != B[bench][i][1]]
        if mism:
            raise ValueError(f"{label_a} vs {label_b}, {name}: {len(mism)} identifiers with different gold answers")
        b = sum(A[bench][i][0] and not B[bench][i][0] for i in ids)
        c = sum(B[bench][i][0] and not A[bench][i][0] for i in ids)
        acc_a = 100 * sum(A[bench][i][0] for i in ids) / len(ids)
        acc_b = 100 * sum(B[bench][i][0] for i in ids) / len(ids)
        rows.append({"contrast": f"{label_a} vs {label_b}", "bench": name, "acc_a": acc_a,
                     "acc_b": acc_b, "a_only": b, "b_only": c, "p": mcnemar(b, c)})
    return rows

def compare_family(pairs):
    """pairs: list of (label_a, path_a, label_b, path_b). Returns rows with Holm within the family."""
    rows = [r for p in pairs for r in compare(*p)]
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a
    return rows

def print_rows(rows, title=None):
    if title:
        print(f"\n=== {title} ({len(rows)} tests) ===")
    print(f"{'contrast':40s} {'bench':8s} {'accA':>6s} {'accB':>6s} {'A>B':>5s} {'B>A':>5s} {'p':>7s} {'p_holm':>7s}")
    for r in rows:
        print(f"{r['contrast']:40s} {r['bench']:8s} {r['acc_a']:6.2f} {r['acc_b']:6.2f} "
              f"{r['a_only']:5d} {r['b_only']:5d} {r['p']:7.4f} {r['p_holm']:7.4f}")

if __name__ == "__main__":
    args = [a.split("=", 1) for a in sys.argv[1:]]
    pairs = [(la, pa, lb, pb) for (la, pa), (lb, pb) in zip(args[0::2], args[1::2])]
    print_rows(compare_family(pairs))
