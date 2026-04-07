"""
plot_paper.py
Generates 6 publication-quality figures using real experimental data.
Run: python -m plots.plot_paper

Figures:
  fig1  — routing quadrant scatter from actual model predictions
  fig2  — sensitivity histogram + real PII type frequencies
  fig3  — theoretical cost surface with actual policy boundaries
  fig4  — policy comparison at matched representative thresholds
  fig5  — cost decomposition from actual cost components
  fig6  — τ_r sensitivity sweep at fixed τ_c
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import Counter

# ── paths ───────────────────────────────────────────────────────────────────
DATA_FILE        = Path("data/documents.jsonl")
RESULTS_CSV      = Path("experiments/results.csv")
PREDICTIONS_JSON = Path("experiments/predictions.json")
OUT_DIR          = Path("plots/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       150,
})

POLICY_COLORS = {
    "autonomous":        "#e74c3c",
    "confidence_only":   "#3498db",
    "privacy_triggered": "#2ecc71",
}
POLICY_LABELS = {
    "autonomous":        "Autonomous AI",
    "confidence_only":   "Confidence-Based L2D",
    "privacy_triggered": "Privacy-Triggered L2D",
}

# Representative thresholds used for single-point comparisons (figs 4-6)
REP_TAU_C = 0.85
REP_TAU_R = 0.5
REP_C_H   = 5.0

HIGH_RISK_PII = {
    "SOCIALNUM", "CREDITCARDNUMBER", "PASSWORD", "ACCOUNTNUM",
    "TAXNUM", "DRIVERLICENSENUM", "IDCARDNUM",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def load_docs() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame()
    records = [json.loads(l) for l in open(DATA_FILE)]
    return pd.DataFrame(records)


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_CSV)


def load_predictions() -> list[dict]:
    if not PREDICTIONS_JSON.exists():
        return []
    return json.loads(open(PREDICTIONS_JSON).read())


def _save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Confidence × Risk scatter from actual model predictions
# ════════════════════════════════════════════════════════════════════════════

def fig1_confidence_risk_scatter(predictions: list[dict]):
    if not predictions:
        print("  [skip] fig1: experiments/predictions.json not found")
        return

    TAU_C, TAU_R = REP_TAU_C, REP_TAU_R

    conf = np.array([p["conf"] for p in predictions])
    risk = np.array([p["risk"] for p in predictions])

    quad = np.where(
        (conf >= TAU_C) & (risk <= TAU_R), "auto",
        np.where(
            (conf >= TAU_C) & (risk > TAU_R), "defer_risk",
            "defer_conf"
        )
    )

    quad_style = {
        "auto":       ("#2ecc71", "o", "Auto-redact (high conf, low risk)", 0.60),
        "defer_risk": ("#e67e22", "^", "Defer — high privacy risk (novel)", 0.85),
        "defer_conf": ("#3498db", "s", "Defer — low confidence",            0.60),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    # Quadrant backgrounds (data coordinates)
    xlo, xhi = 0.0, 1.0
    ylo, yhi = 0.0, 1.0
    ax.add_patch(Rectangle((xlo, ylo), TAU_C - xlo, yhi - ylo,
                            color="#3498db", alpha=0.06, zorder=0))
    ax.add_patch(Rectangle((TAU_C, ylo), xhi - TAU_C, TAU_R - ylo,
                            color="#2ecc71", alpha=0.07, zorder=0))
    ax.add_patch(Rectangle((TAU_C, TAU_R), xhi - TAU_C, yhi - TAU_R,
                            color="#e67e22", alpha=0.08, zorder=0))

    for q, (color, marker, label, alpha) in quad_style.items():
        m = quad == q
        ax.scatter(conf[m], risk[m], c=color, marker=marker,
                   s=30, alpha=alpha, label=f"{label} (n={m.sum()})",
                   linewidths=0)

    ax.axvline(TAU_C, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.axhline(TAU_R, color="black", lw=1.2, ls="--", alpha=0.6)

    pad = 0.03
    xlims = (max(0, conf.min() - pad), min(1, conf.max() + pad))
    ylims = (max(0, risk.min() - pad), min(1, risk.max() + pad))

    ax.text(TAU_C + 0.005, ylims[0] + 0.01, f"τ_c = {TAU_C}",
            fontsize=9, alpha=0.7)
    ax.text(xlims[0] + 0.005, TAU_R + 0.01, f"τ_r = {TAU_R}",
            fontsize=9, alpha=0.7)

    ax.text(0.87, 0.92, "DEFER\n(novel)", fontsize=8.5, color="#e67e22",
            ha="center", va="center", fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#e67e22", alpha=0.7))
    ax.text(0.87, 0.25, "AUTO", fontsize=8.5, color="#27ae60",
            ha="center", va="center", fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#27ae60", alpha=0.7))

    ax.set_xlabel("NER Confidence  c(x)", fontsize=11)
    ax.set_ylabel("Predicted Privacy Risk  r(x)", fontsize=11)
    ax.set_title("Routing Quadrants: Confidence × Privacy Risk", fontsize=13)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.legend(fontsize=8.5, loc="upper left")
    _save(fig, "fig1_confidence_risk_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Dataset distribution: sensitivity scores + real PII types
# ════════════════════════════════════════════════════════════════════════════

def fig2_dataset_distribution(docs: pd.DataFrame):
    if docs.empty or "sensitivity" not in docs.columns:
        print("  [skip] fig2: no document data")
        return

    fig = plt.figure(figsize=(11, 4.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── left: sensitivity score histogram ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    scores = docs["sensitivity"].values
    n_high = int((scores >= 0.5).sum())
    n_low  = int((scores <  0.5).sum())

    ax1.hist(scores[scores < 0.5], bins=18, color="#3498db",
             alpha=0.8, label=f"Low sensitivity (n={n_low})")
    ax1.hist(scores[scores >= 0.5], bins=18, color="#e67e22",
             alpha=0.8, label=f"High sensitivity (n={n_high})")
    ax1.axvline(0.5, color="black", lw=1.3, ls="--", alpha=0.7)
    ax1.text(0.51, ax1.get_ylim()[1] * 0.92, "threshold = 0.5",
             fontsize=8.5, alpha=0.7)
    ax1.set_xlabel("Sensitivity Score  r(x)", fontsize=11)
    ax1.set_ylabel("Document count", fontsize=11)
    ax1.set_title("Sensitivity Score Distribution", fontsize=12)
    ax1.legend(fontsize=9)

    # ── right: PII type frequency bar chart ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    if "pii_types" not in docs.columns:
        print("  [warn] fig2: pii_types column missing — re-run prepare_data.py")
        fig.suptitle("Dataset Analysis", fontsize=14, fontweight="bold", y=1.01)
        _save(fig, "fig2_dataset_distribution.png")
        return

    all_types: list[str] = []
    for row_types in docs["pii_types"]:
        if isinstance(row_types, list):
            all_types.extend(row_types)
    pii_counts = Counter(all_types).most_common(12)

    labels     = [name for name, _ in pii_counts]
    values     = [cnt  for _, cnt  in pii_counts]
    bar_colors = ["#e67e22" if l in HIGH_RISK_PII else "#3498db"
                  for l in labels]

    ax2.barh(labels[::-1], values[::-1], color=bar_colors[::-1],
             alpha=0.85, height=0.65)
    ax2.set_xlabel("Frequency (spans)", fontsize=11)
    ax2.set_title("Top PII Entity Types", fontsize=12)
    ax2.legend(handles=[
        mpatches.Patch(color="#e67e22", label="High-risk PII"),
        mpatches.Patch(color="#3498db", label="Standard PII"),
    ], fontsize=9, loc="lower right")

    fig.suptitle("Dataset Analysis", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, "fig2_dataset_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — Theoretical cost surface with actual policy boundaries
# ════════════════════════════════════════════════════════════════════════════

def fig3_routing_heatmap():
    C_ERR, C_LEAK = 2.0, 50.0
    TAU_C, TAU_R  = REP_TAU_C, REP_TAU_R

    c = np.linspace(0.01, 0.99, 200)
    r = np.linspace(0.01, 0.99, 200)
    C, R = np.meshgrid(c, r)

    cost = (1 - C) * C_ERR + R * C * C_LEAK

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.contourf(C, R, cost, levels=40, cmap="RdYlGn_r", alpha=0.92)
    plt.colorbar(im, ax=ax, label="Approx. per-doc automation cost ($)",
                 shrink=0.88)

    # Confidence-only boundary: vertical line at τ_c (entire y-range)
    ax.axvline(TAU_C, color="white", lw=2.2, ls="--", alpha=0.9,
               label=f"Conf-only boundary (τ_c={TAU_C})")

    # Privacy-triggered boundary: L-shaped
    ax.plot([TAU_C, TAU_C], [0, TAU_R], color="cyan", lw=2.5, alpha=0.9)
    ax.plot([TAU_C, 1.0], [TAU_R, TAU_R], color="cyan", lw=2.5, alpha=0.9,
            label=f"Privacy boundary (τ_r={TAU_R})")

    ax.annotate("High-confidence\ndeferral zone\n(novel)",
                xy=(0.92, 0.75), xytext=(0.55, 0.88),
                fontsize=9, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    ax.text(0.92, 0.25, "AUTO", fontsize=10, color="white",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="none",
                      ec="white", lw=1.5))

    ax.set_xlabel("NER Confidence  c(x)", fontsize=11)
    ax.set_ylabel("Privacy Risk Score  r(x)", fontsize=11)
    ax.set_title(
        "Automation Cost Surface with Policy Boundaries",
        fontsize=13)
    ax.legend(fontsize=9, loc="lower left",
              facecolor="black", labelcolor="white", framealpha=0.5)
    _save(fig, "fig3_routing_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Policy comparison at representative operating points
# ════════════════════════════════════════════════════════════════════════════

def _pick_row(df, policy, tau_c=None, tau_r=None, c_h=REP_C_H):
    mask = (df["policy"] == policy) & (df["C_h"] == c_h)
    if tau_c is not None:
        mask &= (df["tau_c"] == tau_c)
    if tau_r is not None:
        mask &= (df["tau_r"] == tau_r)
    sub = df[mask]
    return sub.iloc[0] if not sub.empty else None


def fig4_policy_comparison(results: pd.DataFrame):
    if results.empty:
        print("  [skip] fig4: no results data")
        return

    rows = {
        "autonomous":        _pick_row(results, "autonomous"),
        "confidence_only":   _pick_row(results, "confidence_only",
                                       tau_c=REP_TAU_C),
        "privacy_triggered": _pick_row(results, "privacy_triggered",
                                       tau_c=REP_TAU_C, tau_r=REP_TAU_R),
    }
    rows = {k: v for k, v in rows.items() if v is not None}
    if len(rows) < 3:
        print("  [warn] fig4: missing data for some policies")
        return

    RATE_METRICS = ["automation_rate", "leakage_rate", "system_f1"]
    RATE_LABELS  = {
        "automation_rate": "Automation\nRate",
        "leakage_rate":    "Leakage\nRate",
        "system_f1":       "System\nF1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [3, 1.5]})

    # ── left: grouped bars for rates / F1 ───────────────────────────────────
    ax = axes[0]
    n_m = len(RATE_METRICS)
    width = 0.22
    x = np.arange(n_m)

    for i, (policy, row) in enumerate(rows.items()):
        offset = (i - 1) * width
        vals = [row[m] for m in RATE_METRICS]
        bars = ax.bar(x + offset, vals, width,
                      label=POLICY_LABELS[policy],
                      color=POLICY_COLORS[policy],
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([RATE_LABELS[m] for m in RATE_METRICS], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Policy Comparison  (τ_c={REP_TAU_C}, τ_r={REP_TAU_R}, "
        f"C_h=${REP_C_H:.0f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)

    # ── right: expected cost bars ───────────────────────────────────────────
    ax2 = axes[1]
    policies = list(rows.keys())
    costs    = [rows[p]["expected_cost"] for p in policies]
    colors   = [POLICY_COLORS[p] for p in policies]
    short    = ["Auto", "Conf", "Priv"]

    bars = ax2.bar(short, costs, color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"${val:.1f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_ylabel("Expected Cost ($)", fontsize=11)
    ax2.set_title("Cost per Document", fontsize=12)

    fig.tight_layout()
    _save(fig, "fig4_policy_comparison_bar.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Cost decomposition from actual cost components
# ════════════════════════════════════════════════════════════════════════════

def fig5_cost_breakdown(results: pd.DataFrame):
    if results.empty:
        print("  [skip] fig5: no results data")
        return
    if "human_cost" not in results.columns:
        print("  [skip] fig5: cost components missing — re-run experiment")
        return

    POLICIES = ["autonomous", "confidence_only", "privacy_triggered"]
    human_c, error_c, leak_c = [], [], []

    for policy in POLICIES:
        if policy == "autonomous":
            row = _pick_row(results, policy)
        elif policy == "confidence_only":
            row = _pick_row(results, policy, tau_c=REP_TAU_C)
        else:
            row = _pick_row(results, policy, tau_c=REP_TAU_C, tau_r=REP_TAU_R)

        if row is None:
            human_c.append(0); error_c.append(0); leak_c.append(0)
            continue
        human_c.append(row["human_cost"])
        error_c.append(row["error_cost"])
        leak_c.append(row["leak_cost"])

    x     = np.arange(len(POLICIES))
    width = 0.45
    labels = [POLICY_LABELS[p] for p in POLICIES]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    b1 = ax.bar(x, human_c, width, label="Human review ($C_h$)",
                color="#3498db", alpha=0.88)
    b2 = ax.bar(x, error_c, width, bottom=human_c,
                label="PII miss ($C_{err}$)", color="#e67e22", alpha=0.88)
    bottom2 = [h + e for h, e in zip(human_c, error_c)]
    b3 = ax.bar(x, leak_c, width, bottom=bottom2,
                label="Privacy leakage ($C_{leak}$)",
                color="#e74c3c", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Expected cost per document ($)", fontsize=11)
    ax.set_title(
        f"Cost Decomposition  (τ_c={REP_TAU_C}, τ_r={REP_TAU_R}, "
        f"C_h=${REP_C_H:.0f})", fontsize=13)
    ax.legend(fontsize=9)

    totals = [h + e + l for h, e, l in zip(human_c, error_c, leak_c)]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 0.15, f"${tot:.2f}", ha="center",
                fontsize=9, fontweight="bold")
    _save(fig, "fig5_cost_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — τ_r sensitivity sweep at fixed τ_c
# ════════════════════════════════════════════════════════════════════════════

def fig6_threshold_sensitivity(results: pd.DataFrame):
    if results.empty:
        print("  [skip] fig6: no results data")
        return

    FIXED_TAU_C = REP_TAU_C
    FIXED_C_H   = REP_C_H

    grp = results[
        (results["policy"] == "privacy_triggered") &
        (results["tau_c"]  == FIXED_TAU_C) &
        (results["C_h"]    == FIXED_C_H)
    ].sort_values("tau_r")

    if grp.empty:
        print(f"  [skip] fig6: no rows at τ_c={FIXED_TAU_C}, C_h={FIXED_C_H}")
        return

    tau_r = grp["tau_r"].values
    leak  = grp["leakage_rate"].values
    auto  = grp["automation_rate"].values
    f1    = grp["system_f1"].values

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax2 = ax1.twinx()
    ax2.grid(False)

    l1, = ax1.plot(tau_r, leak,  color="#e74c3c", lw=2.3,
                   marker="o", markersize=7, label="Leakage rate", zorder=3)
    l3, = ax1.plot(tau_r, f1,    color="#2ecc71", lw=2.0,
                   marker="D", markersize=6, ls="-.",
                   label="System F1", zorder=3)
    l2, = ax2.plot(tau_r, auto,  color="#3498db", lw=2.3,
                   marker="s", markersize=7, ls="--",
                   label="Automation rate", zorder=3)

    # Highlight chosen τ_r
    idx = np.argmin(np.abs(tau_r - REP_TAU_R))
    ax1.scatter([tau_r[idx]], [leak[idx]], s=120, zorder=5,
                color="#e74c3c", edgecolors="black", linewidth=1.5)
    ax2.scatter([tau_r[idx]], [auto[idx]], s=120, zorder=5,
                color="#3498db", edgecolors="black", linewidth=1.5)
    ax1.axvline(tau_r[idx], color="gray", lw=1.2, ls=":", alpha=0.7)
    ax1.text(tau_r[idx] + 0.01, max(leak.max(), f1.max()) * 0.75,
             f"chosen\nτ_r={tau_r[idx]:.1f}", fontsize=8.5, color="gray")

    ax1.set_xlabel("Risk threshold  τ_r", fontsize=11)
    ax1.set_ylabel("Leakage Rate / F1", fontsize=11)
    ax2.set_ylabel("Automation Rate", fontsize=11, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax1.set_title(
        f"Sensitivity Analysis: τ_r sweep  "
        f"(fixed τ_c={FIXED_TAU_C}, C_h=${FIXED_C_H:.0f})", fontsize=12)

    ax1.legend(handles=[l1, l2, l3],
               labels=[l.get_label() for l in [l1, l2, l3]],
               fontsize=9, loc="center left")
    _save(fig, "fig6_threshold_sensitivity.png")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    docs        = load_docs()
    results     = load_results()
    predictions = load_predictions()

    if docs.empty:
        print("  [warn] data/documents.jsonl not found")
    if results.empty:
        print("  [warn] experiments/results.csv not found")
    if not predictions:
        print("  [warn] experiments/predictions.json not found")

    print("\nGenerating figures...")
    fig1_confidence_risk_scatter(predictions)
    fig2_dataset_distribution(docs)
    fig3_routing_heatmap()
    fig4_policy_comparison(results)
    fig5_cost_breakdown(results)
    fig6_threshold_sensitivity(results)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Figure placement:")
    print("  Fig 1 → Introduction / Motivation")
    print("  Fig 2 → Experimental Setup")
    print("  Fig 3 → Methodology")
    print("  Fig 4 → Results")
    print("  Fig 5 → Results / Cost Analysis")
    print("  Fig 6 → Discussion / Ablation")


if __name__ == "__main__":
    main()
