"""Figures for the CAiSE paper. Values copied from the evaluation logs (per-seed where available)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 2: accuracy per condition, seeds as dots ----
conds = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
m4 = [[60.96, 60.64, 60.96], [62.22, 61.35, 61.12], [61.12, 62.06, 61.04], [61.12, 61.82, 60.88], [61.12, 61.82, 61.04],
      [61.04, 62.22, 61.19], [61.19, 61.90, 60.80], [60.88, 62.45, 61.35], [60.88]]
m4ms = [None, None, None, None, None, (61.48, 0.64), (61.30, 0.56), (61.56, 0.81), None]
m5 = [[51.77, 51.14, 50.35], [50.51, 53.42, 51.37], [52.32, 54.20, 52.00], [52.08, 53.73, 52.40], [52.71, 53.81, 52.40],
      [50.90, 53.26, 51.69], [52.16, 53.89, 51.92], [50.75, 53.18, 51.53], [49.73]]
m5ms = [None, None, None, None, None, (51.95, 1.20), (52.66, 1.07), (51.82, 1.24), None]
col = ["#7f7f7f", "#7f7f7f", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728", "#d62728", "#d62728", "#9467bd"]
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
for ax, vals, ms, title, ref in [(axes[0], m4, m4ms, "(a) MedQA, four options", 58.68), (axes[1], m5, m5ms, "(b) MedQA, five options", 52.63)]:
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
