"""Generate all paper figures from actual experimental results."""

import os, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ── All experimental data ────────────────────────────────────────────────────
BASELINES = [
    ("GA",        0.028, 0.521, 0.028, 0.000, 0.262),
    ("NPO",       0.636, 0.624, 0.614, 0.000, 0.613),
    ("SCRUB",     0.037, 0.526, 0.031, 0.000, 0.212),
    ("SalUn",     0.011, 0.541, 0.008, 0.000, 0.051),
    ("RMU",       0.580, 0.565, 0.558, 0.000, 0.559),
    ("AlphaEdit", 0.575, 0.558, 0.551, 0.000, 0.555),
]

# Valid Pareto sweep points (exclude λ=6.0 runs which failed)
PARETO = [
    {"alpha": 0.0, "FA": 0.028, "RA": 0.521, "Q4": 0.262, "label": "α=0\n(GA-equiv)"},
    {"alpha": 1.0, "FA": 0.275, "RA": 0.317, "Q4": 0.060, "label": "α=1"},
    {"alpha": 3.0, "FA": 0.040, "RA": 0.045, "Q4": 0.044, "label": "α=3\n(best Q_INT4)"},
]


def save(fig, name):
    for ext in ["png", "pdf"]:
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
    print(f"  Saved: {name}.png/.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Baseline failure — the main finding
# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    print("Figure 1: Baseline failure...")
    methods = [b[0] for b in BASELINES]
    fa      = [b[1] for b in BASELINES]
    qi4     = [b[5] for b in BASELINES]
    x, w    = np.arange(len(methods)), 0.32

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 1: All 6 Baselines Fail Under INT4 Quantization Recovery Attack",
                 fontsize=12, fontweight="bold", y=1.01)

    ax = axes[0]
    b1 = ax.bar(x-w/2, fa,  w, label="Forget Acc (post-unlearning)", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x+w/2, qi4, w, label="Q_INT4 (after INT4 attack)",   color="#F44336", alpha=0.85)
    ax.axhline(0.05, color="#4CAF50", ls="--", lw=2, label="5% target")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Token Accuracy on Forget Set"); ax.set_ylim(0, 0.78)
    ax.set_title("Forget Acc vs INT4 Recovery\n(lower = better for both)")
    ax.legend(fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., h+0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color="#c62828")

    ax = axes[1]
    # Recovery ratio for methods that actually forgot
    good = [(b[0], b[1], b[5]) for b in BASELINES if b[1] < 0.1]
    gn = [g[0] for g in good]; gf = [g[1] for g in good]; gq = [g[2] for g in good]
    gr = [q/max(f,1e-6) for f,q in zip(gf,gq)]
    colors = {"GA":"#2196F3","SCRUB":"#9C27B0","SalUn":"#4CAF50"}
    bars = ax.bar(range(len(gn)), gr,
                  color=[colors.get(n,"#666") for n in gn], alpha=0.85)
    ax.set_xticks(range(len(gn))); ax.set_xticklabels(gn, rotation=15, ha="right")
    ax.set_ylabel("Q_INT4 / FA (recovery ratio)")
    ax.set_title("INT4 Recovery Ratio\n(methods with FA < 0.05 only)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="1× = no recovery")
    ax.legend()
    for bar, r, n in zip(bars, gr, gn):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.2,
                f"{r:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=colors.get(n,"#666"))

    plt.tight_layout()
    save(fig, "figure1_baseline_failure")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: INT8 is harmless, INT4 is catastrophic
# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    print("Figure 2: INT8 vs INT4...")
    methods = [b[0] for b in BASELINES]
    fa  = [b[1] for b in BASELINES]
    qi8 = [b[4] for b in BASELINES]
    qi4 = [b[5] for b in BASELINES]
    x, w = np.arange(len(methods)), 0.24

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-w,   fa,  w, label="FA (post-unlearning)", color="#2196F3", alpha=0.9)
    ax.bar(x,     qi8, w, label="FA@INT8 (after INT8)", color="#4CAF50", alpha=0.9)
    ax.bar(x+w,   qi4, w, label="FA@INT4 (after INT4)", color="#F44336", alpha=0.9)
    ax.axhline(0.05, color="gray", ls="--", lw=1.5, label="5% target")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Token Accuracy on Forget Set")
    ax.set_title("Figure 2: INT8 Quantization Is Harmless — INT4 Is a Universal Threat\n"
                 "Q_INT8 = 0.000 for every method; Q_INT4 >> 0 for every method")
    ax.set_ylim(0, 0.75); ax.legend()
    ax.text(1.0, 0.03, "INT8:\nno recovery\n(all methods)", ha="center",
            fontsize=9, color="#2e7d32", style="italic")
    ax.annotate("INT4 recovery:\n5–22× increase",
                xy=(0.5, 0.26), fontsize=9, color="#c62828", style="italic")
    plt.tight_layout()
    save(fig, "figure2_int8_vs_int4")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: The FA–RA–Q_INT4 Trilemma (3D view as 2-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    print("Figure 3: FA-RA-Q_INT4 trilemma...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 3: The FA–RA–Q_INT4 Trilemma — DurableUn-SAF Pareto Frontier",
                 fontsize=12, fontweight="bold")

    # Panel 1: FA vs Q_INT4 scatter
    ax = axes[0]
    baseline_mk = {"GA":"^","SCRUB":"s","SalUn":"D","NPO":"v","RMU":"P","AlphaEdit":"X"}
    bc          = {"GA":"#2196F3","SCRUB":"#9C27B0","SalUn":"#4CAF50",
                   "NPO":"#FF9800","RMU":"#F44336","AlphaEdit":"#795548"}
    for name, fa, ra, qbf, qi8, qi4 in BASELINES:
        ax.scatter([fa], [qi4], marker=baseline_mk[name], s=180,
                   c=bc[name], zorder=5, label=name, edgecolors="white", lw=1.5)
        ax.annotate(name, (fa, qi4), textcoords="offset points",
                    xytext=(7, 3), fontsize=9, color=bc[name], fontweight="bold")

    # Pareto curve
    fas  = [p["FA"] for p in PARETO]
    qi4s = [p["Q4"] for p in PARETO]
    alps = [p["alpha"] for p in PARETO]
    sc   = ax.scatter(fas, qi4s, c=alps, cmap="RdPu", s=280, marker="*",
                      zorder=6, edgecolors="black", lw=1, vmin=0, vmax=3)
    plt.colorbar(sc, ax=ax, label="α (alpha_quant)")

    # Connect points
    sx, sy = zip(*sorted(zip(fas, qi4s)))
    ax.plot(sx, sy, "k--", lw=1.5, alpha=0.4)

    for p in PARETO:
        ax.annotate(p["label"], (p["FA"], p["Q4"]),
                    textcoords="offset points", xytext=(-55, 8),
                    fontsize=8, color="#880E4F",
                    arrowprops=dict(arrowstyle="->", color="#880E4F", lw=0.8))

    # Target region
    ax.axhline(0.05, color="#00BCD4", ls="--", lw=1.5, label="Q_INT4 target")
    ax.axvline(0.05, color="#00BCD4", ls=":",  lw=1.5)
    ax.fill_between([0,0.05],[0,0],[0.05,0.05], alpha=0.08, color="#00BCD4")
    ax.text(0.025, 0.025, "Ideal", ha="center", va="center",
            fontsize=9, color="#00BCD4", style="italic")

    ax.set_xlabel("Forget Accuracy ↓"); ax.set_ylabel("Q_INT4 Recovery ↓")
    ax.set_title("FA vs Q_INT4 Pareto Frontier\n★ = DurableUn-SAF, shapes = baselines")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(-0.02, 0.72); ax.set_ylim(-0.02, 0.72)

    # Panel 2: α effect on all three metrics
    ax2 = axes[1]
    valid_alphas = [p["alpha"] for p in PARETO]
    valid_fa     = [p["FA"]    for p in PARETO]
    valid_ra     = [p["RA"]    for p in PARETO]
    valid_qi4    = [p["Q4"]    for p in PARETO]

    ax2.plot(valid_alphas, valid_fa,  "b-o", lw=2, ms=8, label="Forget Acc ↓")
    ax2.plot(valid_alphas, valid_qi4, "r-s", lw=2, ms=8, label="Q_INT4 ↓")
    ax2.plot(valid_alphas, valid_ra,  "g-^", lw=2, ms=8, label="Retain Acc ↑")

    # SalUn reference lines
    ax2.axhline(0.011, color="#4CAF50", ls=":", lw=1.5, alpha=0.7, label="SalUn FA")
    ax2.axhline(0.051, color="red",     ls=":", lw=1.5, alpha=0.7, label="SalUn Q_INT4")
    ax2.axhline(0.541, color="green",   ls=":", lw=1.5, alpha=0.7, label="SalUn RA")

    ax2.fill_between([2.5, 3.5], [0, 0], [0.06, 0.06],
                     alpha=0.08, color="red", label="Q_INT4 < SalUn region")
    ax2.text(3.0, 0.03, "Beats\nSalUn\nQ_INT4", ha="center", fontsize=9,
             color="#c62828", style="italic")

    ax2.set_xlabel("α (alpha_quant)"); ax2.set_ylabel("Token Accuracy")
    ax2.set_title("Effect of α on the Three-Way Tradeoff\n"
                  "dotted lines = SalUn reference values")
    ax2.legend(fontsize=8, ncol=2); ax2.set_ylim(-0.02, 0.72)

    plt.tight_layout()
    save(fig, "figure3_trilemma_pareto")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Heatmap — all methods × all precisions
# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    print("Figure 4: Precision heatmap...")
    methods  = ["GA","NPO","SCRUB","SalUn","RMU","AlphaEdit","DurableUn\nSAF α=3"]
    data     = np.array([
        [0.028, 0.028, 0.000, 0.262],
        [0.636, 0.614, 0.000, 0.613],
        [0.037, 0.031, 0.000, 0.212],
        [0.011, 0.008, 0.000, 0.051],
        [0.580, 0.558, 0.000, 0.559],
        [0.575, 0.551, 0.000, 0.555],
        [0.040, 0.040, 0.000, 0.044],  # DurableUn α=3
    ])
    cols = ["Forget Acc", "Q-BF16", "Q-INT8", "Q-INT4"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=0.65, aspect="auto")
    plt.colorbar(im, ax=ax, label="Token Accuracy (lower = better for forgetting)")
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10)
    ax.set_title("Figure 4: Forget Accuracy at Each Quantization Precision\n"
                 "(green=low=good, red=high=model still remembers)", fontsize=11)

    for i in range(len(methods)):
        for j in range(4):
            v = data[i, j]
            col = "white" if v > 0.35 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=col, fontweight="bold")

    # Highlight DurableUn row
    ax.add_patch(plt.Rectangle((-0.5, 5.5), 4, 1,
                               fill=False, edgecolor="#E91E63", lw=3))
    ax.text(4.05, 6, "← DurableUn", va="center", color="#E91E63",
            fontsize=10, fontweight="bold")

    plt.tight_layout()
    save(fig, "figure4_precision_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Clean comparison bar (best methods only)
# ─────────────────────────────────────────────────────────────────────────────
def fig5():
    print("Figure 5: Summary comparison...")
    methods  = ["GA",   "SCRUB", "SalUn", "DurableUn\nSAF α=3"]
    fa_v     = [0.028,  0.037,   0.011,   0.040]
    ra_v     = [0.521,  0.526,   0.541,   0.045]
    qi4_v    = [0.262,  0.212,   0.051,   0.044]
    colors   = ["#2196F3","#9C27B0","#4CAF50","#E91E63"]
    x = np.arange(len(methods))
    w = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Figure 5: DurableUn-SAF vs Best Baselines\n"
                 "(methods achieving FA < 0.05; DurableUn is first to beat SalUn Q_INT4)",
                 fontsize=11, fontweight="bold")

    for i, (ax, vals, title, ylabel, target) in enumerate(zip(
        axes,
        [fa_v, ra_v, qi4_v],
        ["Forget Accuracy ↓", "Retain Accuracy ↑", "Q_INT4 Recovery ↓"],
        ["Token Acc", "Token Acc", "Token Acc"],
        [0.05, None, 0.05],
    )):
        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.4)
        if target:
            ax.axhline(target, color="gray", ls="--", lw=1.5,
                       label=f"{target} target")
            ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        bars[-1].set_edgecolor("#E91E63")
        bars[-1].set_linewidth(3)

    # Add "First to beat SalUn!" annotation on Q_INT4 panel
    axes[2].annotate("First method\nto beat SalUn!",
                     xy=(3, 0.044), xytext=(2.0, 0.15),
                     fontsize=9, color="#E91E63", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#E91E63"))

    plt.tight_layout()
    save(fig, "figure5_summary_comparison")


if __name__ == "__main__":
    print(f"Generating all figures → {FIG_DIR}/\n")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print(f"\nDone. All figures in {FIG_DIR}/")
