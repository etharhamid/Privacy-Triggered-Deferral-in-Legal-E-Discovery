"""
plot_pareto.py
Reads experiments/results.csv and generates:
  - pareto_workload_vs_leakage.png
  - pareto_cost_vs_f1.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("experiments/results.csv")
OUT_DIR = Path("plots")

COLORS = {
    "autonomous":        "#e74c3c",
    "confidence_only":   "#3498db",
    "privacy_triggered": "#2ecc71",
    "joint_cost":        "#9b59b6",
}
LABELS = {
    "autonomous":        "Autonomous AI",
    "confidence_only":   "Confidence-Based L2D",
    "privacy_triggered": "Privacy-Triggered L2D",
    "joint_cost":        "Joint Cost",
}
MARKERS = {
    "autonomous":        "D",
    "confidence_only":   "o",
    "privacy_triggered": "s",
    "joint_cost":        "^",
}
LINESTYLES = {
    "autonomous":        "-",
    "confidence_only":   "-",
    "privacy_triggered": "--",
    "joint_cost":        "-.",
}


def _pareto_front(pts, x_col, y_col, x_higher_better=True, y_higher_better=False):
    """Keep only Pareto-optimal rows (non-dominated points)."""
    pts = pts.sort_values(x_col, ascending=not x_higher_better).reset_index(drop=True)
    best_y = float("inf") if not y_higher_better else float("-inf")
    keep = []
    for i, row in pts.iterrows():
        y = row[y_col]
        if (not y_higher_better and y <= best_y) or (y_higher_better and y >= best_y):
            keep.append(i)
            best_y = y
    return pts.loc[keep]


def plot_leakage_vs_automation(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for policy in ["autonomous", "confidence_only", "privacy_triggered"]:
        grp = df[df["policy"] == policy].copy()
        if grp.empty:
            continue

        if policy == "autonomous":
            pt = grp.iloc[0]
            ax.scatter(
                pt["automation_rate"], pt["leakage_rate"],
                marker=MARKERS[policy], s=120, color=COLORS[policy],
                label=LABELS[policy], zorder=5, edgecolors="white", linewidth=0.8,
            )
            continue

        front = _pareto_front(grp, "automation_rate", "leakage_rate",
                              x_higher_better=True, y_higher_better=False)
        front = front.sort_values("automation_rate")

        ax.plot(
            front["automation_rate"], front["leakage_rate"],
            marker=MARKERS[policy],
            label=LABELS[policy],
            color=COLORS[policy],
            linewidth=2,
            linestyle=LINESTYLES[policy],
            markersize=7,
            alpha=0.9,
        )

    ax.set_xlabel("Automation Rate (→ less human work)", fontsize=11)
    ax.set_ylabel("True Leakage Rate (↓ safer)", fontsize=11)
    ax.set_title("Pareto Frontier: Automation vs. True Privacy Leakage", fontsize=13)
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=10, title_fontsize=8)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "pareto_workload_vs_leakage.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


def plot_cost_vs_f1(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for policy in ["autonomous", "confidence_only", "privacy_triggered", "joint_cost"]:
        grp = df[df["policy"] == policy].copy()
        if grp.empty:
            continue

        if policy == "autonomous":
            pt = grp.iloc[0]
            ax.scatter(
                pt["expected_cost"], pt["system_f1"],
                marker=MARKERS[policy], s=120, color=COLORS[policy],
                label=LABELS[policy], zorder=5, edgecolors="white", linewidth=0.8,
            )
            continue

        pareto = (
            grp.groupby("C_h")[["expected_cost", "system_f1"]]
            .apply(lambda g: g.loc[g["system_f1"].idxmax()])
            .reset_index(drop=True)
            .sort_values("expected_cost")
        )

        ax.plot(
            pareto["expected_cost"], pareto["system_f1"],
            marker=MARKERS[policy],
            label=LABELS[policy],
            color=COLORS[policy],
            linewidth=2,
            linestyle=LINESTYLES[policy],
            markersize=7,
        )

    ax.set_xlabel("Expected Cost per Document (← cheaper)", fontsize=11)
    ax.set_ylabel("System F1 Score (↑ better)", fontsize=11)
    ax.set_title("Cost vs. Redaction Quality ($C_h$ sweep)", fontsize=13)
    ax.legend(fontsize=10)

    out = OUT_DIR / "pareto_cost_vs_f1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RESULTS)
    if "seed" in df.columns:
        group = ["policy", "tau_c", "tau_r", "C_h"]
        nums = [c for c in df.columns if c not in group + ["seed"]]
        agg = df.groupby(group, dropna=False)[nums].agg("mean").reset_index()
        df = agg
    plot_leakage_vs_automation(df)
    plot_cost_vs_f1(df)


if __name__ == "__main__":
    main()