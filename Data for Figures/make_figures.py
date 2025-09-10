# make_figures.py
# -*- coding: utf-8 -*-
"""
Load precomputed results from /results and reproduce the figures EXACTLY as in the original notebook.
This script ONLY reads the saved CSV/JSON files and plots, matching rcParams, labels, colors, and geometry.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- GLOBAL STYLE ----
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 4,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.markersize': 10
})

RESULTS_DIR = Path("results")
FIG_DPI = 300

def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None

def fig1A_accuracy_vs_temp():
    df = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    extrema = _load_json(RESULTS_DIR / "figure1A_extrema.json")

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    sns.lineplot(data=df, x="temperature", y="accuracy", marker='o')

    # Flitz labels above points
    for _, row in df.iterrows():
        lbl = "" if "flitz_labels" not in row or pd.isna(row["flitz_labels"]) else str(row["flitz_labels"])
        plt.text(
            x=row["temperature"],
            y=row["accuracy"] + 0.03,
            s=lbl,
            ha="center",
            fontsize=18
        )

    # 50% threshold
    plt.axhline(0.5, color='red', linestyle='--')
    plt.text(
        x=df['temperature'].max() + 0.05,
        y=0.48,
        s='50% accuracy threshold',
        color='red',
        fontsize=15,
        ha='right',
        va='top'
    )

    # Local maxima/minima (indices saved by run_analysis.py)
    temps_np = df['temperature'].to_numpy()
    accs_np  = df['accuracy'].to_numpy()
    max_idx = (extrema or {}).get("local_max_indices", [])
    min_idx = (extrema or {}).get("local_min_indices", [])

    for i in max_idx:
        temp, acc = temps_np[i], accs_np[i]
        plt.scatter(temp, acc, color="green", zorder=5)
        plt.text(
            temp, acc + 0.06,
            f"Local max\n({temp:.2f}, {acc:.2f})",
            fontsize=15, color='green', ha='center', va='bottom'
        )

    for i in min_idx:
        temp, acc = temps_np[i], accs_np[i]
        plt.scatter(temp, acc, color="orange", zorder=5)
        plt.text(
            temp, acc - 0.035,
            f"Local min\n({temp:.2f}, {acc:.2f})",
            fontsize=15, color='orange', ha='center', va='top'
        )

    # (A) in bold (same transform)
    plt.text(
        x=0.04, y=0.96, s='(A)',
        transform=plt.gcf().transFigure,
        fontsize=25, fontweight='bold',
        ha='left', va='top'
    )

    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.savefig("figure1A_accuracy_vs_temp.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def fig1B_deviation_from_chance():
    df = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    x = df['temperature'].to_numpy()
    y = df['accuracy'].to_numpy()
    abs_dev = np.abs(y - 0.5)

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    plt.plot(x, abs_dev, marker='o', linestyle='-', color='purple', linewidth=4)
    plt.axhline(0, linestyle='--', color='red')

    for i in range(len(x)):
        if abs_dev[i] > 0.35:
            y_offset, va = -0.03, 'top'
        else:
            y_offset, va = 0.02, 'bottom'
        plt.text(
            x[i], abs_dev[i] + y_offset,
            f'{abs_dev[i]:.2f}', ha='center', va=va, fontsize=15, color='black'
        )

    plt.text(
        x=0.04, y=0.96, s='(B)',
        transform=plt.gcf().transFigure,
        fontsize=25, fontweight='bold',
        ha='left', va='top'
    )
    plt.text(
        x=x.max() + 0.05, y=0.01,
        s='50% accuracy threshold',
        color='red', fontsize=15, ha='right', va='bottom'
    )
    plt.xlabel("Temperature")
    plt.ylabel("|Accuracy − 0.50|")
    plt.xticks(x)
    plt.ylim(-0.02, 0.45)
    plt.tight_layout()
    plt.savefig("figure1B_deviation_from_chance.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def fig2_avg_rating_vs_temp():
    df = pd.read_csv(RESULTS_DIR / "ratings_by_temp.csv").sort_values("temperature")
    extrema = _load_json(RESULTS_DIR / "figure2_extrema.json")

    x = df['temperature'].astype(float).to_numpy()
    y = df['avg_rating'].astype(float).to_numpy()
    yerr = df['std_dev'].astype(float).to_numpy()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    sns.lineplot(x=x, y=y, marker='o', label='Average rating')
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='blue', label='±1 Std dev')

    # Extrema annotations
    max_idx = (extrema or {}).get("local_max_indices", [])
    min_idx = (extrema or {}).get("local_min_indices", [])

    for i in max_idx:
        plt.scatter(x[i], y[i], color='green', zorder=5)
        plt.text(
            x[i], y[i] + 0.2,
            f"Local max\n({x[i]:.2f}, {y[i]:.2f})",
            fontsize=15, color='green', ha='center', va='bottom'
        )
    for i in min_idx:
        plt.scatter(x[i], y[i], color='orange', zorder=5)
        plt.text(
            x[i], y[i] - 0.3,
            f"Local min\n({x[i]:.2f}, {y[i]:.2f})",
            fontsize=15, color='orange', ha='center', va='top'
        )

    plt.xlabel('Temperature')
    plt.ylabel('Average rating')
    plt.xlim(0, 2.0)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig("figure2_avg_rating_vs_temp.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def fig3_logistic_fit_by_author():
    """
    EXACT reproducibility: do NOT refit here.
    Use the precomputed binned means/SE and logistic predictions saved by run_analysis.py
    """
    bins_stats = pd.read_csv(RESULTS_DIR / "fig3_binned_means_se.csv")
    preds = pd.read_csv(RESULTS_DIR / "logistic_fit_predictions.csv")

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})

    colors = {"AI": "orange", "Human": "blue"}

    # Plot mean ± SE per author
    for author_type in ["AI", "Human"]:
        s = bins_stats[bins_stats["author_type"] == author_type]
        x_pts = s["rating_bin_mid"].astype(float).to_numpy()
        y_pts = s["mean_correct"].to_numpy()
        y_err = s["sem_correct"].to_numpy()

        plt.errorbar(
            x_pts, y_pts,
            yerr=y_err,
            label=f"{author_type} (mean ± SE)",
            fmt='o',
            capsize=5,
            color=colors[author_type]
        )

    # Overlay logistic predictions (precomputed)
    for author_type in ["AI", "Human"]:
        s = preds[preds["Author"] == author_type]
        x_line = s["Rating"].to_numpy()
        y_line = s["Predicted_Probability"].to_numpy()
        plt.plot(
            x_line, y_line,
            linestyle='--', linewidth=4,
            label=f"{author_type} (logistic fit)",
            color=colors[author_type]
        )

    plt.xlabel("Rating (1–10)")
    plt.ylabel("1=correct, 0=incorrect")
    plt.ylim(0.1, 1.3)
    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    plt.tight_layout()
    plt.savefig("figure3_logistic_fit_by_author.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def fig4_human_flitz_accuracy():
    """
    Robust to column types and missing labels; enforces the exact xtick order.
    If any required flitz labels are missing, raise a clear error telling you to re-run run_analysis.py.
    """
    df = pd.read_csv(RESULTS_DIR / "accuracy_by_flitz_human.csv")
    if "flitz_label" not in df.columns:
        # Backward compatibility: derive it if missing
        if "flitz_id" in df.columns:
            df["flitz_label"] = df["flitz_id"].astype(str)
        else:
            raise KeyError("Expected 'flitz_label' (or 'flitz_id') in accuracy_by_flitz_human.csv")

    # Normalize labels as strings with no whitespace
    df["flitz_label"] = df["flitz_label"].astype(str).str.strip()

    xtick_order = ['1','2','12','8','6','7','11','13','16']

    # Reindex in the specified order; this won't crash if some are missing
    df_idx = df.set_index('flitz_label')
    df_sorted = df_idx.reindex(xtick_order)

    # If any are missing, fail fast with a clear message
    missing = [lbl for lbl in xtick_order if lbl not in df_idx.index]
    if missing:
        raise KeyError(
            f"Missing human flitz labels in accuracy_by_flitz_human.csv: {missing}. "
            f"Re-run run_analysis.py to regenerate results."
        )

    accs = df_sorted['accuracy'].to_numpy()
    labels = np.array(xtick_order)
    x_pos = np.arange(len(labels))

    # Compute local extrema here (purely for text placements)
    from scipy.signal import argrelextrema
    max_idx = argrelextrema(accs, np.greater)[0].tolist()
    min_idx = argrelextrema(accs, np.less)[0].tolist()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x_pos, accs, marker='o', linestyle='-', color='orange')

    for idx in max_idx:
        plt.scatter(x_pos[idx], accs[idx], color='green', zorder=5)
        plt.text(
            x_pos[idx], accs[idx] + 0.03,
            f"Local max\n({labels[idx]}, {accs[idx]:.2f})",
            color='green', fontsize=15, ha='center', va='bottom'
        )

    for idx in min_idx:
        plt.scatter(x_pos[idx], accs[idx], color='orange', zorder=5)
        plt.text(
            x_pos[idx], accs[idx] - 0.04,
            f"Local min\n({labels[idx]}, {accs[idx]:.2f})",
            color='orange', fontsize=15, ha='center', va='top'
        )

    plt.axhline(0.5, color='red', linestyle='--')
    plt.text(
        x_pos[-1] + 0.25, 0.47,
        '50% accuracy threshold',
        color='red', fontsize=15, ha='right', va='bottom'
    )

    plt.xticks(x_pos, labels.tolist())
    plt.xlabel('Flitz number')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 0.87)
    plt.tight_layout()
    plt.savefig("figure4_human_flitz_accuracy.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def fig5_overlay_human_ai_accuracy():
    overlay = pd.read_csv(RESULTS_DIR / "figure5_overlay_data.csv")

    # Enforce explicit order
    temps = overlay['temperature'].astype(float).to_list()
    ai_acc = overlay['ai_acc'].astype(float).to_list()
    human_acc = overlay['human_acc'].astype(float).to_list()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    sns.lineplot(x=temps, y=ai_acc, marker='o', label='AI Flitzes', linewidth=4, color='orange')
    sns.lineplot(x=temps, y=human_acc, marker='s', label='Human Flitzes', linewidth=4, color='blue')

    plt.axhline(0.5, color='red', linestyle='--', linewidth=4)
    plt.text(
        max(temps) + 0.05, 0.48,
        '50% threshold',
        color='red', fontsize=15, ha='right', va='top'
    )

    plt.text(
        x=0.04, y=0.96, s='(A)',
        transform=plt.gcf().transFigure,
        fontsize=25, fontweight='bold',
        ha='left', va='top'
    )

    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.ylim(0.3, 1.05)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("figure5_overlay_human_ai_accuracy.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # FIGURE REPRODUCTION PIPELINE (reads only from /results)
    fig1A_accuracy_vs_temp()
    fig1B_deviation_from_chance()
    fig2_avg_rating_vs_temp()
    fig3_logistic_fit_by_author()
    fig4_human_flitz_accuracy()
    fig5_overlay_human_ai_accuracy()