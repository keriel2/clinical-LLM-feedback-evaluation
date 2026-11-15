"""fmaca_table.py: original FMACA-CE study, mean +/- SD over seeds, exact McNemar vs static fine-tuning, plus LaTeX."""
import os, json, math
ROOT = "."
BENCH = ["medqa", "biasmedqa", "pubmedqa"]
SEEDS = [42, 43, 44]
REF = "static_sft"
ROWS = [
    ("base", "Base model (no fine-tuning)", "ablation_baselines/base_s0/eval_results.json"),
    ("static_sft", "Static fine-tuning", "ablation_baselines/static_sft_s{s}/static_sft/eval_results.json"),
    ("static_coord", "Static, through the Coordinator", "ablation_baselines/static_coord_s{s}/static_coord/eval_results.json"),
    ("quality_only", "Quality-based selection", "ablation_baselines/quality_only_s{s}/quality_only/eval_results.json"),
    ("error_curriculum", "Error-feedback selection", "ablation_baselines/error_curriculum_s{s}/error_curriculum/eval_results.json"),
    ("typed", "FMACA-CE (typed negatives)", "ablation_ce_neg/typed_s{s}/fmaca/eval_results.json"),
    ("random", "FMACA-CE, random negatives", "ablation_ce_neg/random_s{s}/fmaca/eval_results.json"),
    ("shuffled_types", "FMACA-CE, shuffled error types", "ablation_ce_neg/shuffled_types_s{s}/fmaca/eval_results.json"),
]
def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan"))
runs, missing = {}, []
for key, _, tpl in ROWS:
    for s in ([0] if key == "base" else SEEDS):
        p = os.path.join(ROOT, tpl.format(s=s))
        if os.path.exists(p): runs[(key, s)] = json.load(open(p))
        else: missing.append(p)
stats = {}
print("\nAccuracy (%), mean +/- SD; [k/n] = seeds with p<0.05 vs static fine-tuning (exact McNemar), + better / - worse")
print(f"{'condition':34s} {'seeds':9s}" + "".join(f"{b:>24s}" for b in BENCH))
for key, label, _ in ROWS:
    seeds = [s for s in ([0] if key == "base" else SEEDS) if (key, s) in runs]
    if not seeds: print(f"{label:34s} missing"); continue
    line = f"{label:34s} {','.join(map(str, seeds)):9s}"
    for b in BENCH:
        accs = [100 * runs[(key, s)][b]["accuracy"] for s in seeds if b in runs[(key, s)]]
        m, sd = mean_sd(accs)
        sig = better = worse = tested = 0
        for s in seeds:
            r = runs.get((REF, s if key != "base" else 42))
            if key == REF or not r or b not in r or b not in runs[(key, s)]: continue
            x, y = r[b]["per_example"], runs[(key, s)][b]["per_example"]
            if len(x) != len(y): continue
            bb = sum(1 for u, v in zip(x, y) if u["correct"] and not v["correct"])
            cc = sum(1 for u, v in zip(x, y) if v["correct"] and not u["correct"])
            tested += 1
            if mcnemar(bb, cc) < 0.05: sig += 1; better += cc > bb; worse += bb > cc
        stats[(key, b)] = (m, sd, len(accs), sig, tested)
        tag = "" if key == REF or not tested else f" [{sig}/{tested}{'+' * better}{'-' * worse}]"
        line += f"{(f'{m:.2f}' + (f' +/- {sd:.2f}' if not math.isnan(sd) else '') + tag):>24s}"
    print(line)
if missing:
    print("\nMissing runs:"); [print("  ", p) for p in missing]
print("\n% ---- LaTeX (booktabs) ----")
print("\\begin{table}[t]\n\\centering\n\\caption{Accuracy (\\%) of the original FMACA-CE study, mean $\\pm$ SD over three seeds. "
      "$^{*}$: exact McNemar $p<0.05$ against static fine-tuning in at least two of three seeds.}")
print("\\label{tab:fmaca}\n\\begin{tabular}{lccc}\n\\toprule\nCondition & MedQA & BiasMedQA & PubMedQA \\\\\n\\midrule")
for key, label, _ in ROWS:
    cells = []
    for b in BENCH:
        if (key, b) not in stats: cells.append("--"); continue
        m, sd, n, sig, tested = stats[(key, b)]
        cells.append(f"{m:.2f}" + (f" $\\pm$ {sd:.2f}" if n > 1 else "") + ("$^{*}$" if key != REF and tested and sig >= 2 else ""))
    print(f"{label} & " + " & ".join(cells) + " \\\\")
    if key in ("base", "error_curriculum"): print("\\midrule")
print("\\bottomrule\n\\end{tabular}\n\\end{table}")
