"""Figures 2 and 3 of the paper.
Figure 2 is computed from results/ (per-seed accuracy difference from C3).
Figure 3 uses values from the seed-42 training logs (contrastive-step losses), which are not
included because of their size: share of the 1,446 contrastive steps with loss > 1e-3
(typed 26%, random negatives 6 of 1,446 = 0.4%, DPO 78%) and the DPO loss every 200 steps."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAMES = {"C1": "C1_static", "C2": "C2_static_batch4", "C3": "C3_loop_feedback_off", "C4": "C4_quality_selection",
         "C5": "C5_error_feedback_selection", "C6": "C6_full_typed", "C7": "C7_full_random_negatives",
         "C8": "C8_full_random_error_types", "C9": "C9_full_dpo"}
BENCH = ["medqa", "biasmedqa", "pubmedqa"]          # 'biasmedqa' = five-option MedQA (no added bias)
def acc(c, s):
    f = os.path.join(R, f"{NAMES[c]}_s{s}", "eval_results.json")
    if not os.path.exists(f): return None
    d = json.load(open(f)); return [100 * d[b]["accuracy"] for b in BENCH]

# ---- Figure 2 ----
rows = ["C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
labels = {"C1": "C1 static", "C2": "C2 static, matched updates", "C4": "C4 quality selection",
          "C5": "C5 error-feedback selection", "C6": "C6 full system", "C7": "C7 random negatives",
          "C8": "C8 random error types", "C9": "C9 DPO (one seed)"}
col = {"C1": "#7f7f7f", "C2": "#7f7f7f", "C4": "#2ca02c", "C5": "#2ca02c", "C6": "#d62728",
       "C7": "#d62728", "C8": "#d62728", "C9": "#9467bd"}
titles = ["(a) MedQA, four options", "(b) MedQA, five options", "(c) PubMedQA"]
def diffs(c, j):
    out = []
    for s in (42, 43, 44):
        a, b = acc(c, s), acc("C3", s)
        if a and b: out.append(a[j] - b[j])
    return out
fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.4), sharey=True)
for j, ax in enumerate(axes):
    for i, c in enumerate(rows):
        d = diffs(c, j); y = len(rows) - 1 - i
        ax.scatter(d, [y] * len(d), s=9, color=col[c], zorder=3, alpha=.85)
        ax.plot([np.mean(d)] * 2, [y - .28, y + .28], color=col[c], lw=1.6)
    ax.axvline(0, color="black", lw=.7, ls=":")
    c1 = np.mean(diffs("C1", j))
    ax.axvline(c1, color="#7f7f7f", lw=.8, ls="--", zorder=1)
    ax.text(c1, len(rows) - 0.35, "C1 mean ", fontsize=5.5, color="#555555", ha="right", va="bottom")
    ax.text(0, len(rows) - 0.35, " C3", fontsize=5.5, color="black", ha="left", va="bottom")
    ax.set_title(titles[j], fontsize=6.5, loc="left", pad=9); ax.set_xlim(-3.2, 1.6); ax.tick_params(labelsize=6)
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([labels[c] for c in rows[::-1]], fontsize=6)
fig.supxlabel("Accuracy difference from C3 in the same seed (percentage points)", fontsize=6.5)
fig.tight_layout(w_pad=0.6); fig.savefig("fig2_accuracy.pdf"); plt.close(fig)

# ---- Figure 3 ----
fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9))
ax = axes[0]; names = ["typed,\nmargin", "random,\nmargin", "typed,\nDPO"]; active = [26, 0.4, 78]
ax.barh(range(3), active, color=["#d62728", "#d62728", "#9467bd"])
for i, a in enumerate(active): ax.text(a + 2, i, f"{a:.1f}%" if a < 1 else f"{a:g}%", va="center", fontsize=6.5)
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("Steps with loss > $10^{-3}$ (%)", fontsize=6.5); ax.set_title("(a) Active contrastive steps", fontsize=7, loc="left")
ax = axes[1]; steps = [1, 201, 401, 601, 801, 1001, 1201, 1401]; loss = [0.6494, 0.5370, 0.0177, 0.0002, 0, 0, 0.0058, 0.0741]
ax.plot(steps, loss, "-o", color="#9467bd", ms=3); ax.axhline(np.log(2), color="#999999", ls=":", lw=0.8)
ax.text(1420, np.log(2) + 0.02, "log 2", fontsize=6, ha="right"); ax.set_ylim(-0.03, 0.78)
ax.set_xlabel("Contrastive step", fontsize=6.5); ax.set_ylabel("DPO loss", fontsize=6.5)
ax.set_title("(b) DPO loss on 110 recycled pairs", fontsize=7, loc="left")
fig.tight_layout(); fig.savefig("fig3_feedback.pdf"); plt.close(fig)
print("ok")
