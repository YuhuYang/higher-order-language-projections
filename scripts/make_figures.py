#!/usr/bin/env python
"""Generate manuscript Figures 2--5 from analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#666666",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


def fig2_layer_similarity(tables_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "layer_similarity.csv")
    metrics = [
        ("projection_edge_jaccard", "Edge Jaccard"),
        ("dep_proj_degree_spearman", "Degree rho"),
        ("pagerank_spearman", "PageRank rho"),
        ("neighborhood_jaccard_mean", "Neighborhood J"),
        ("community_nmi", "Community NMI"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    vals = [df[c].mean() for c, _ in metrics]
    err = [df[c].std(ddof=1) for c, _ in metrics]
    x = np.arange(len(metrics))
    ax.bar(x, vals, yerr=err, color=[OKABE["sky"], OKABE["green"], OKABE["green"], OKABE["orange"], OKABE["purple"]], capsize=2, linewidth=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean across languages")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=35, ha="right")
    ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, linestyle=":")
    add_panel_label(ax, "A")

    ax = axes[1]
    ax.scatter(df["dep_avg_clustering"], df["proj_avg_clustering"], s=28, color=OKABE["vermillion"], alpha=0.8, edgecolor="white", linewidth=0.4)
    lim = [0, max(df["proj_avg_clustering"].max(), df["dep_avg_clustering"].max()) * 1.08]
    ax.plot(lim, lim, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Dependency clustering")
    ax.set_ylabel("Projection clustering")
    ax.set_title("Projection increases closure")
    add_panel_label(ax, "B")
    fig.tight_layout()
    save(fig, "fig2_layer_similarity", output_dir)


def fig3_clique_taxonomy(tables_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "clique_taxonomy.csv").sort_values("assembled_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x, df["within_ratio"], color=OKABE["blue"], label="Within-hyperedge")
    ax.bar(x, df["assembled_ratio"], bottom=df["within_ratio"], color=OKABE["orange"], label="Assembled")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Triangle fraction")
    ax.set_xlabel("Treebanks sorted by assembled fraction")
    ax.set_xticks([])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    add_panel_label(ax, "A")

    ax = axes[1]
    parts = [df["within_ratio"].mean(), df["assembled_ratio"].mean(), df["mixed_ratio"].mean()]
    labels = ["Within", "Assembled", "Mixed"]
    colors = [OKABE["blue"], OKABE["orange"], OKABE["purple"]]
    x2 = np.array([0.0, 1.0, 2.55])
    ax.bar(x2, parts, width=0.72, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 3.15)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean fraction")
    ax.set_title("Two classification dimensions")
    ax.axvline(1.75, color="#AAAAAA", linewidth=0.8, linestyle=":")
    ax.text(0.5, 0.94, "Boundary status", ha="center", fontsize=7, fontweight="bold")
    ax.text(2.55, 0.94, "Edge composition", ha="center", fontsize=7, fontweight="bold")
    ax.text(2.55, 0.88, "non-exclusive", ha="center", fontsize=6.5, color=OKABE["gray"])
    for x, v in zip(x2, parts):
        ax.text(x, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
    add_panel_label(ax, "B")
    fig.tight_layout()
    save(fig, "fig3_clique_taxonomy", output_dir)


def fig4_recoverability_null(tables_dir: Path, output_dir: Path) -> None:
    rec = pd.read_csv(tables_dir / "recoverability.csv")
    null = pd.read_csv(tables_dir / "null_zscores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    metrics = ["recover_precision", "recover_recall", "recover_f1", "boundary_error_mean"]
    labels = ["Precision", "Recall", "F1", "Boundary\nerror"]
    vals = [rec[m].mean() for m in metrics]
    err = [rec[m].std(ddof=1) for m in metrics]
    ax.bar(np.arange(len(metrics)), vals, yerr=err, color=[OKABE["blue"], OKABE["green"], OKABE["orange"], OKABE["vermillion"]], capsize=2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean across languages")
    ax.set_title("Projection recoverability")
    add_panel_label(ax, "A")

    ax = axes[1]
    keep = [
        "unique_hyperedge_count",
        "role_entropy",
        "hyperdegree_entropy",
        "incidence_entropy_mean",
        "repeated_role_rate",
        "overlap_mean",
    ]
    rows = []
    for metric in keep:
        vals = null.loc[null["metric"] == metric, "z"].dropna()
        if vals.empty:
            rows.append(np.nan)
            continue
        direction = 1 if vals.mean() >= 0 else -1
        rows.append(float(((vals * direction) > 2).mean()))
    y = np.arange(len(keep))
    ax.barh(y, rows, color=OKABE["purple"])
    ax.axvline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([
        "Unique hyperedges",
        "Role entropy",
        "Hyperdegree entropy",
        "Incidence entropy",
        "Repeated roles",
        "Overlap mean",
    ])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fraction of languages with directional |z| > 2")
    ax.set_title("Consistent deviation from null")
    for yi, val in enumerate(rows):
        if np.isfinite(val):
            ax.text(min(val + 0.025, 0.98), yi, f"{val:.2f}", va="center", fontsize=7)
    add_panel_label(ax, "B")
    fig.tight_layout()
    save(fig, "fig4_recoverability_null", output_dir)


def fig5_prediction(tables_dir: Path, output_dir: Path) -> None:
    pred = pd.read_csv(tables_dir / "prediction_scores.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = np.arange(len(pred))
    width = 0.38
    ax.bar(x - width / 2, pred["auc"], width=width, color=OKABE["blue"], label="AUC")
    ax.bar(x + width / 2, pred["f1"], width=width, color=OKABE["orange"], label="F1")
    ax.set_ylim(0.6, 0.86)
    ax.set_ylabel("Cross-validated score")
    ax.set_xticks(x)
    ax.set_xticklabels(pred["model"], rotation=25, ha="right")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Boundary prediction gains from higher-order features")
    for i, row in pred.iterrows():
        ax.text(i - width / 2, row["auc"] + 0.006, f"{row['auc']:.2f}", ha="center", fontsize=7)
        ax.text(i + width / 2, row["f1"] + 0.006, f"{row['f1']:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_prediction_gain", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figures 2--5 from analysis tables."
    )
    parser.add_argument(
        "--tables-dir",
        required=True,
        type=Path,
        help="directory containing the CSV tables from run_analysis.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for PNG and PDF figures",
    )
    args = parser.parse_args()

    setup_style()
    fig2_layer_similarity(args.tables_dir, args.output_dir)
    fig3_clique_taxonomy(args.tables_dir, args.output_dir)
    fig4_recoverability_null(args.tables_dir, args.output_dir)
    fig5_prediction(args.tables_dir, args.output_dir)
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
