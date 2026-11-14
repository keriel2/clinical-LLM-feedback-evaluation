"""Figures 2 and 3 of the paper. Figure 2 is computed from results/; Figure 3 uses values from the training logs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 2: accuracy per condition, seeds as dots (read from results/) ----
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAMES = ["C1_static", "C2_static_batch4", "C3_loop_feedback_off", "C4_quality_selection",
         "C5_error_feedback_selection", "C6_full_typed", "C7_full_random_negatives",
         "C8_full_random_error_types", "C9_full_dpo"]
conds = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
def acc(name, bench):
    out = []
    for s in (42, 43, 44):
        f = os.path.join(R, f"{name}_s{s}", "eval_results.json")
        if os.path.exists(f):
            out.append(100 * json.load(open(f))[bench]["accuracy"])
    return out
m4 = [acc(n, "medqa") for n in NAMES]
m5 = [acc(n, "biasmedqa") for n in NAMES]          # 'biasmedqa' = five-option MedQA
base = json.load(open(os.path.join(R, "C0_base", "eval_results.json")))
ref4, ref5 = 100 * base["medqa"]["accuracy"], 100 * base["biasmedqa"]["accuracy"]
m4ms = m5ms = [None] * len(conds)
col = ["#7f7f7f", "#7f7f7f", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728", "#d62728", "#d62728", "#9467bd"]
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
for ax, vals, ms, title, ref in [(axes[0], m4, m4ms, "(a) MedQA, four options", ref4), (axes[1], m5, m5ms, "(b) MedQA, five options", ref5)]:
    x = np.arange(len(conds))
    for i in x:
        if vals[i] is not None:
            v = vals[i]; mu = np.mean(v)
            ax.bar(i, mu, 0.6, color=col[i], alpha=0.25)
            ax.scatter([i] * len(v), v, color=col[i], s=7, zorder=3)
        else:
            mu, sd = ms[i]
            ax.bar(i, mu, 0.6, color=col[i], alpha=0.25)
            ax.errorbar(i, mu, yerr=sd, color=col[i], capsize=3, fmt="o", ms=3)
    ax.axhline(ref, color="black", lw=0.7, ls=":")
    ax.text(-0.45, ref + 0.08, "no fine-tuning", fontsize=6, ha="left", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=6.5)
    ax.set_ylim(ref - 4.5 if "five" in title else 58, 63.5 if "four" in title else 55.5)
    ax.set_ylabel("Accuracy (%)"); ax.set_title(title, fontsize=7, loc="left")
    ax.axvspan(2.5, 4.5, color="#2ca02c", alpha=0.05); ax.axvspan(4.5, 8.5, color="#d62728", alpha=0.04)
fig.tight_layout(); fig.savefig("fig2_accuracy.pdf")

# ---- Figure 3: feedback starvation ----
# Values transcribed from the seed-42 training logs (contrastive-step losses), which are not
# included in this repository because of their size: share of the 1,446 contrastive steps with
# loss > 1e-3, and the DPO loss sampled every 200 contrastive steps.
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axes[0]
names = ["typed,\nmargin", "random,\nmargin", "typed,\nDPO"]
active = [26, 0, 78]
ax.barh(names, active, color=["#d62728", "#ff9896", "#9467bd"])
for i, a in enumerate(active):
    ax.text(a + 2, i, f"{a}%", va="center", fontsize=6.5)
ax.set_xlim(0, 100); ax.set_xlabel(r"Steps with loss > $10^{-3}$ (%)")
ax.set_title("(a) Active contrastive steps", fontsize=7, loc="left")
ax.invert_yaxis()
ax = axes[1]
steps = [1, 201, 401, 601, 801, 1001, 1201, 1401]
loss = [0.6494, 0.5370, 0.0177, 0.0002, 0.0, 0.0, 0.0058, 0.0741]
ax.plot(steps, loss, "o-", color="#9467bd", ms=3)
ax.axhline(np.log(2), color="#999999", ls=":", lw=0.8); ax.text(1420, np.log(2) + 0.02, "log 2", fontsize=6, ha="right")
ax.set_xlabel("Contrastive step"); ax.set_ylabel("DPO loss")
ax.set_title("(b) DPO loss on 110 recycled pairs", fontsize=7, loc="left")
ax.set_ylim(-0.03, 0.78)
fig.tight_layout(); fig.savefig("fig3_feedback.pdf")
print("ok")
