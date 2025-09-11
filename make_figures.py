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


# --- HELPER ---
def add_panel_label_outside(ax, label, fig=None, pad=0.012, xpad=0.0, fontsize=25):
    """
    Place a bold panel label just ABOVE the axes (outside the plotting area).
    pad/xpad are in figure coordinates.
    """
    if fig is None:
        fig = ax.figure
    # Make sure axes positions are finalized before we read them
    fig.canvas.draw()
    bbox = ax.get_position()  # in figure coords
    x = bbox.x0 + xpad
    y = bbox.y1 + pad
    fig.text(x, y, label, fontsize=fontsize, fontweight='bold', ha='left', va='bottom')

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

# --- NEW: (A) |Accuracy-0.50| vs Temp  +  (B) Logistic/ratings subplot with OUTSIDE labels ---
def fig_AB_subplots_accuracy_dev_and_logistic_OUTSIDE():
    import numpy as np
    import pandas as pd
    from pathlib import Path

    RESULTS_DIR = Path("results")
    FIG_DPI = 300

    # (A) data
    acc = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    xA = acc["temperature"].to_numpy()
    yA = acc["accuracy"].to_numpy()
    abs_dev = np.abs(yA - 0.5)

    # (B) data (precomputed only)
    bins_stats = pd.read_csv(RESULTS_DIR / "fig3_binned_means_se.csv")
    preds = pd.read_csv(RESULTS_DIR / "logistic_fit_predictions.csv")
    colors = {"AI": "orange", "Human": "blue"}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 6))

    # ---------- (A) ----------
    axA.plot(xA, abs_dev, marker='o', linestyle='-', color='purple', linewidth=4)
    axA.axhline(0, linestyle='--', color='red')
    for i in range(len(xA)):
        y_offset, va = (-0.03, 'top') if abs_dev[i] > 0.35 else (0.02, 'bottom')
        axA.text(xA[i], abs_dev[i] + y_offset, f'{abs_dev[i]:.2f}', ha='center', va=va, fontsize=15, color='black')
    axA.set_xlabel("Temperature")
    axA.set_ylabel("|Accuracy − 0.50|")
    axA.set_xticks(xA)
    axA.set_ylim(-0.02, 0.45)
    axA.text(xA.max() + 0.05, 0.01, '50% accuracy threshold', color='red', fontsize=15, ha='right', va='bottom')

    # ---------- (B) ----------
    for author in ["AI", "Human"]:
        s = bins_stats[bins_stats["author_type"] == author]
        axB.errorbar(
            s["rating_bin_mid"].astype(float).to_numpy(),
            s["mean_correct"].to_numpy(),
            yerr=s["sem_correct"].to_numpy(),
            fmt='o', capsize=5, color=colors[author],
            label=f"{author} (mean ± SE)"
        )
    for author in ["AI", "Human"]:
        s = preds[preds["Author"] == author]
        axB.plot(
            s["Rating"].to_numpy(),
            s["Predicted_Probability"].to_numpy(),
            linestyle='--', linewidth=4, color=colors[author],
            label=f"{author} (logistic fit)"
        )
    axB.set_xlabel("Rating (1–10)")
    axB.set_ylabel("1=correct, 0=incorrect")
    axB.set_ylim(0.1, 1.3)
    axB.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    # Layout first, THEN add outside labels
    fig.tight_layout()
    fig.canvas.draw()
    add_panel_label_outside(axA, "(A)", fig=fig)
    add_panel_label_outside(axB, "(B)", fig=fig)

    fig.savefig("figure_AB_outside_accuracydev_and_logistic.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

# --- NEW: (A) Temp vs Accuracy  +  (B) Temp vs Avg Rating  with OUTSIDE labels ---
def fig_AB_subplots_acc_vs_temp_AND_rating_vs_temp_OUTSIDE():
    import numpy as np
    import pandas as pd
    from pathlib import Path

    RESULTS_DIR = Path("results")
    FIG_DPI = 300

    # Left panel data: accuracy vs temp
    acc = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    try:
        extrema1 = json.loads((RESULTS_DIR / "figure1A_extrema.json").read_text())
        max_idx = extrema1.get("local_max_indices", [])
        min_idx = extrema1.get("local_min_indices", [])
    except FileNotFoundError:
        max_idx = []
        min_idx = []

    xL = acc["temperature"].to_numpy()
    yL = acc["accuracy"].to_numpy()
    flitz_labels = acc.get("flitz_labels", pd.Series([""]*len(acc))).astype(str).tolist()

    # Right panel data: avg rating vs temp
    r = pd.read_csv(RESULTS_DIR / "ratings_by_temp.csv").sort_values("temperature")
    try:
        extrema2 = json.loads((RESULTS_DIR / "figure2_extrema.json").read_text())
        max_idx_r = extrema2.get("local_max_indices", [])
        min_idx_r = extrema2.get("local_min_indices", [])
    except FileNotFoundError:
        max_idx_r = []
        min_idx_r = []

    xR = r["temperature"].astype(float).to_numpy()
    yR = r["avg_rating"].astype(float).to_numpy()
    yerrR = r["std_dev"].astype(float).to_numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 6))

    # ---------- (A) Temp vs Accuracy ----------
    sns.lineplot(ax=axL, x=xL, y=yL, marker='o')
    for xi, yi, lbl in zip(xL, yL, flitz_labels):
        if lbl and lbl != "nan":
            axL.text(xi, yi + 0.03, lbl, ha='center', fontsize=18)
    axL.axhline(0.5, color='red', linestyle='--')
    if len(xL) > 0:
        axL.text(xL.max() + 0.05, 0.48, '50% accuracy threshold', color='red', fontsize=15, ha='right', va='top')

    # annotate extrema (precomputed)
    for i in max_idx:
        axL.scatter(xL[i], yL[i], color="green", zorder=5)
        axL.text(xL[i], yL[i] + 0.06, f"Local max\n({xL[i]:.2f}, {yL[i]:.2f})",
                 fontsize=15, color='green', ha='center', va='bottom')
    for i in min_idx:
        axL.scatter(xL[i], yL[i], color="orange", zorder=5)
        axL.text(xL[i], yL[i] - 0.035, f"Local min\n({xL[i]:.2f}, {yL[i]:.2f})",
                 fontsize=15, color='orange', ha='center', va='top')

    axL.set_xlabel("Temperature")
    axL.set_ylabel("Accuracy")
    axL.set_ylim(0.3, 1.05)

    # ---------- (B) Temp vs Avg Rating ----------
    sns.lineplot(ax=axR, x=xR, y=yR, marker='o', label='Average rating')
    axR.fill_between(xR, yR - yerrR, yR + yerrR, alpha=0.2, color='blue', label='±1 Std dev')

    for i in max_idx_r:
        axR.scatter(xR[i], yR[i], color='green', zorder=5)
        axR.text(xR[i], yR[i] + 0.2, f"Local max\n({xR[i]:.2f}, {yR[i]:.2f})",
                 fontsize=15, color='green', ha='center', va='bottom')
    for i in min_idx_r:
        axR.scatter(xR[i], yR[i], color='orange', zorder=5)
        axR.text(xR[i], yR[i] - 0.3, f"Local min\n({xR[i]:.2f}, {yR[i]:.2f})",
                 fontsize=15, color='orange', ha='center', va='top')

    axR.set_xlabel("Temperature")
    axR.set_ylabel("Average rating")
    axR.set_xlim(0, 2.0)
    axR.legend(fontsize=14, loc='upper right')

    # Layout first, THEN outside labels
    fig.tight_layout()
    fig.canvas.draw()
    add_panel_label_outside(axL, "(A)", fig=fig)
    add_panel_label_outside(axR, "(B)", fig=fig)

    fig.savefig("figure_AB_outside_acc_vs_temp_and_rating_vs_temp.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

def fig_ABCD_outside_acc_rating_consecutive_ref0():
    """
    2x2 subplots with OUTSIDE panel labels:
      (A) Temperature vs Accuracy                [top-left]
      (B) Temperature vs Average rating          [top-right]
      (C) Temperature (midpoint) vs Distance     [bottom-left]  (consecutive temperatures)
      (D) Temperature vs Wasserstein Distance    [bottom-right] (reference T=0.00)

    Uses your existing figure styling (global rcParams) and precomputed results where possible.
    For (C) and (D) it will read from results/* if present, else fall back to Desktop Excel paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    RESULTS_DIR = Path("results")
    FIG_DPI = 300

    # ---------- helpers ----------
    def _first_existing(paths):
        for p in paths:
            if Path(p).exists():
                return Path(p)
        return None

    def _read_any(p):
        p = Path(p)
        if p.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return pd.read_csv(p)

    # ---------- (A) Temp vs Accuracy (like fig1A) ----------
    acc = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    xA = acc["temperature"].to_numpy()
    yA = acc["accuracy"].to_numpy()
    flitz_labels = acc.get("flitz_labels", pd.Series([""] * len(acc))).astype(str).tolist()

    # ---------- (B) Temp vs Avg Rating (like fig2) ----------
    ratings = pd.read_csv(RESULTS_DIR / "ratings_by_temp.csv").sort_values("temperature")
    xB = ratings["temperature"].astype(float).to_numpy()
    yB = ratings["avg_rating"].astype(float).to_numpy()
    yB_err = ratings["std_dev"].astype(float).to_numpy()

    # Canonical temp ticks (used for (C))
    temp_ticks = acc["temperature"].astype(float).dropna().unique()
    temp_ticks = np.array(sorted(temp_ticks))

    # ---------- (C) Consecutive distances ----------
    # Prefer results/*, else Desktop xlsx
    consecutive_path = _first_existing([
        RESULTS_DIR / "wasserstein_distance_consecutive_temps.csv",
        RESULTS_DIR / "wasserstein_distance_consecutive_temps.xlsx",
        Path.home() / "Desktop" / "wasserstein_distance_consecutive_temps.xlsx",
    ])
    if consecutive_path is None:
        raise FileNotFoundError(
            "Missing consecutive distance data. Expected one of:\n"
            " - results/wasserstein_distance_consecutive_temps.csv\n"
            " - results/wasserstein_distance_consecutive_temps.xlsx\n"
            " - ~/Desktop/wasserstein_distance_consecutive_temps.xlsx"
        )
    consecutive_df = _read_any(consecutive_path)

    # Robust column handling
    if "temperature_midpoint" in consecutive_df.columns:
        xC = consecutive_df["temperature_midpoint"].astype(float).to_numpy()
    elif {"temperature_left", "temperature_right"}.issubset(consecutive_df.columns):
        xC = (consecutive_df["temperature_left"].astype(float).to_numpy() +
              consecutive_df["temperature_right"].astype(float).to_numpy()) / 2.0
    else:
        # Fallback: try a generic 'temperature' column
        xC = consecutive_df["temperature"].astype(float).to_numpy()

    # y for (C)
    yC_col = "wasserstein_distance" if "wasserstein_distance" in consecutive_df.columns else "distance"
    yC = consecutive_df[yC_col].astype(float).to_numpy()

    # ---------- (D) Distance from T=0.00 ----------
    reference_path = _first_existing([
        RESULTS_DIR / "wasserstein_distance_reference_point.csv",
        RESULTS_DIR / "wasserstein_distance_reference_point.xlsx",
        Path.home() / "Desktop" / "wasserstein_distance_reference_point.xlsx",
    ])
    if reference_path is None:
        raise FileNotFoundError(
            "Missing reference-point distance data. Expected one of:\n"
            " - results/wasserstein_distance_reference_point.csv\n"
            " - results/wasserstein_distance_reference_point.xlsx\n"
            " - ~/Desktop/wasserstein_distance_reference_point.xlsx"
        )
    reference_df = _read_any(reference_path)

    # Robust columns for (D)
    if "temperature" not in reference_df.columns:
        # try to infer
        cand = [c for c in reference_df.columns if "temp" in c.lower()][0]
    else:
        cand = "temperature"
    xD = reference_df[cand].astype(float).to_numpy()

    mean_col = "mean_distance" if "mean_distance" in reference_df.columns else "mean"
    std_col  = "std_distance"  if "std_distance"  in reference_df.columns else ("std" if "std" in reference_df.columns else None)

    yD = reference_df[mean_col].astype(float).to_numpy()
    yD_err = reference_df[std_col].astype(float).to_numpy() if std_col else None

    # ---------- build 2x2 layout ----------
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axA, axB = axs[0]
    axC, axD = axs[1]

    # ===================== (A) Temperature vs Accuracy =====================
    sns.lineplot(ax=axA, x=xA, y=yA, marker='o')
    # flitz labels above points (same as your template)
    for xi, yi, lbl in zip(xA, yA, flitz_labels):
        if lbl and lbl != "nan":
            axA.text(xi, yi + 0.03, lbl, ha='center', fontsize=18)
    # 50% threshold
    axA.axhline(0.5, color='red', linestyle='--')
    if len(xA) > 0:
        axA.text(xA.max() + 0.05, 0.48, '50% accuracy threshold', color='red', fontsize=15, ha='right', va='top')
    axA.set_xlabel("Temperature")
    axA.set_ylabel("Accuracy")
    axA.set_ylim(0.3, 1.05)

    # ===================== (B) Temperature vs Average rating =====================
    sns.lineplot(ax=axB, x=xB, y=yB, marker='o', label='Average rating')
    axB.fill_between(xB, yB - yB_err, yB + yB_err, alpha=0.2, color='blue', label='±1 Std dev')
    axB.set_xlabel("Temperature")
    axB.set_ylabel("Average rating")
    axB.set_xlim(0.0, 2.0)
    axB.legend(fontsize=14, loc='upper right')

    # ===================== (C) Consecutive temperature distance =====================
    sns.lineplot(ax=axC, x=xC, y=yC, marker='o')
    # show canonical temperature ticks for reference (no rotation to match template)
    if temp_ticks.size:
        axC.set_xticks(temp_ticks)
    axC.set_xlabel("Temperature")
    axC.set_ylabel("Distance (consecutive temperatures)")
    axC.set_xlim(0.0, 2.0)

    # ===================== (D) Distance from T=0.00 =====================
    sns.lineplot(ax=axD, x=xD, y=yD, marker='o', label='Mean distance')
    if yD_err is not None:
        axD.fill_between(xD, yD - yD_err, yD + yD_err, alpha=0.2, color='blue', label='±1 SD')
    axD.set_xlabel("Temperature")
    axD.set_ylabel("Wasserstein distance from T=0.00")
    axD.set_xlim(0.0, 2.0)
    axD.legend(loc='lower right', fontsize=14)

    # ---------- layout + OUTSIDE panel labels ----------
    fig.tight_layout()
    fig.canvas.draw()
    # uses the helper you already added earlier
    add_panel_label_outside(axA, "(A)", fig=fig)
    add_panel_label_outside(axB, "(B)", fig=fig)
    add_panel_label_outside(axC, "(C)", fig=fig)
    add_panel_label_outside(axD, "(D)", fig=fig)

    fig.savefig("figure_ABCD_outside_2x2.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # FIGURE REPRODUCTION PIPELINE (reads only from /results)
    fig1A_accuracy_vs_temp()
    fig1B_deviation_from_chance()
    fig2_avg_rating_vs_temp()
    fig3_logistic_fit_by_author()
    fig4_human_flitz_accuracy()
    fig5_overlay_human_ai_accuracy()

    fig_AB_subplots_accuracy_dev_and_logistic_OUTSIDE()
    fig_AB_subplots_acc_vs_temp_AND_rating_vs_temp_OUTSIDE()
    fig_ABCD_outside_acc_rating_consecutive_ref0()