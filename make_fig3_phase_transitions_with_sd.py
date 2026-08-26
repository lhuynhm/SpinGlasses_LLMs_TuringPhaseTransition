# make_fig3_phase_transitions_with_sd.py
# -*- coding: utf-8 -*-
"""
Re-create fig3_phase_transitions.png (originally produced by
`make_figures_consistency_phasetransitions_data.fig_ABCD_outside_acc_rating_consecutive_ref0`,
which wrote figure_ABCD_outside_2x2.png) with ONE change:

    panel (A) Temperature vs Accuracy now carries a shaded spread band,
    drawn exactly the way panel (B) draws its "+-1 Std dev" band
    (fill_between, alpha=0.2, color='blue', legend entry).

Panels (C) and (D) are byte-for-byte the same code as the original.  Panel (B) is
too, except that `band_b` can switch its shaded band from the original SD of the
individual ratings to the SE of the mean rating, so (A) and (B) can be made to
show the same estimator.

Two band flavours are produced, because "1 standard deviation" is ambiguous for a
binary outcome:

  band="sem"  ->  +-1 SE of the mean accuracy = sd/sqrt(n) ~ +-0.05.
                  Fits inside the original y-limits, so nothing else moves.
  band="sd"   ->  +-1 SD of the individual correct/incorrect responses
                  (the literal analogue of panel (B), which shows the SD of the
                  individual ratings).  For a Bernoulli variable that is
                  sqrt(p(1-p)) ~ 0.5, so the y-limits have to be widened.

Usage:
    python make_fig3_phase_transitions_with_sd.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- GLOBAL STYLE (identical to make_figures_consistency_phasetransitions_data.py) ----
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 4,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.markersize': 10
})

RESULTS_DIR = Path("results_fig")          # where run_statistical_analysis.py wrote its CSVs
DATA_DIR = Path("Data for Figures")        # where the Wasserstein workbooks live
FIG_DPI = 300


def add_panel_label_outside(ax, label, fig=None, pad=0.012, xpad=0.0, fontsize=25):
    """Place a bold panel label just ABOVE the axes (outside the plotting area)."""
    if fig is None:
        fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_position()
    fig.text(bbox.x0 + xpad, bbox.y1 + pad, label,
             fontsize=fontsize, fontweight='bold', ha='left', va='bottom')


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


def accuracy_spread_by_temp():
    """
    Per-temperature spread of the accuracy in panel (A), from the per-response
    tidy table.  Same estimator panel (B) uses for the ratings (pandas .std(),
    ddof=1), just applied to the 0/1 `correct` column, plus its SE.
    """
    tidy = pd.read_csv(RESULTS_DIR / "tidy_df.csv")
    sub = tidy.dropna(subset=["temperature"])
    g = sub.groupby("temperature")["correct"].agg(["mean", "std", "count"])
    g = g.rename(columns={"mean": "accuracy", "std": "std_dev", "count": "n"})
    g["sem"] = g["std_dev"] / np.sqrt(g["n"])
    return g.reset_index().sort_values("temperature")


def make_figure(band="sem", band_b="sd", out_png=None):
    """
    band   : spread shown in panel (A)  -- "sem" or "sd"
    band_b : spread shown in panel (B)  -- "sd" (the original) or "sem"
    """
    assert band in ("sem", "sd")
    assert band_b in ("sem", "sd")

    # ---------- (A) Temp vs Accuracy ----------
    acc = pd.read_csv(RESULTS_DIR / "accuracy_by_temp_labeled.csv").sort_values("temperature")
    xA = acc["temperature"].to_numpy()
    yA = acc["accuracy"].to_numpy()
    flitz_labels = acc.get("flitz_labels", pd.Series([""] * len(acc))).astype(str).tolist()

    spread = accuracy_spread_by_temp()
    # align the spread table to the temperatures actually plotted
    spread = (acc[["temperature"]]
              .merge(spread, on="temperature", how="left")
              .sort_values("temperature"))
    yA_err = spread["sem"].to_numpy() if band == "sem" else spread["std_dev"].to_numpy()
    band_label = "±1 SE" if band == "sem" else "±1 Std dev"

    # ---------- (B) Temp vs Avg Rating ----------
    ratings = pd.read_csv(RESULTS_DIR / "ratings_by_temp.csv").sort_values("temperature")
    xB = ratings["temperature"].astype(float).to_numpy()
    yB = ratings["avg_rating"].astype(float).to_numpy()
    yB_sd = ratings["std_dev"].astype(float).to_numpy()
    yB_sem = yB_sd / np.sqrt(ratings["count"].astype(float).to_numpy())
    yB_err = yB_sd if band_b == "sd" else yB_sem
    band_b_label = "±1 Std dev" if band_b == "sd" else "±1 SE"

    temp_ticks = np.array(sorted(acc["temperature"].astype(float).dropna().unique()))

    # ---------- (C) Consecutive distances ----------
    consecutive_path = _first_existing([
        RESULTS_DIR / "wasserstein_distance_consecutive_temps.csv",
        RESULTS_DIR / "wasserstein_distance_consecutive_temps.xlsx",
        DATA_DIR / "wasserstein_distance_consecutive_temps.xlsx",
        Path.home() / "Desktop" / "wasserstein_distance_consecutive_temps.xlsx",
    ])
    if consecutive_path is None:
        raise FileNotFoundError("Missing consecutive distance data.")
    consecutive_df = _read_any(consecutive_path)

    if "temperature_midpoint" in consecutive_df.columns:
        xC = consecutive_df["temperature_midpoint"].astype(float).to_numpy()
    elif {"temperature_left", "temperature_right"}.issubset(consecutive_df.columns):
        xC = (consecutive_df["temperature_left"].astype(float).to_numpy() +
              consecutive_df["temperature_right"].astype(float).to_numpy()) / 2.0
    else:
        xC = consecutive_df["temperature"].astype(float).to_numpy()

    yC_col = "wasserstein_distance" if "wasserstein_distance" in consecutive_df.columns else "distance"
    yC = consecutive_df[yC_col].astype(float).to_numpy()

    # ---------- (D) Distance from T=0.00 ----------
    reference_path = _first_existing([
        RESULTS_DIR / "wasserstein_distance_reference_point.csv",
        RESULTS_DIR / "wasserstein_distance_reference_point.xlsx",
        DATA_DIR / "wasserstein_distance_reference_point.xlsx",
        Path.home() / "Desktop" / "wasserstein_distance_reference_point.xlsx",
    ])
    if reference_path is None:
        raise FileNotFoundError("Missing reference-point distance data.")
    reference_df = _read_any(reference_path)

    if "temperature" not in reference_df.columns:
        cand = [c for c in reference_df.columns if "temp" in c.lower()][0]
    else:
        cand = "temperature"
    xD = reference_df[cand].astype(float).to_numpy()

    mean_col = "mean_distance" if "mean_distance" in reference_df.columns else "mean"
    std_col = "std_distance" if "std_distance" in reference_df.columns else ("std" if "std" in reference_df.columns else None)
    yD = reference_df[mean_col].astype(float).to_numpy()
    yD_err = reference_df[std_col].astype(float).to_numpy() if std_col else None

    # ---------- build 2x2 layout ----------
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axA, axB = axs[0]
    axC, axD = axs[1]

    # ===================== (A) Temperature vs Accuracy =====================
    sns.lineplot(ax=axA, x=xA, y=yA, marker='o', label='Accuracy')
    axA.fill_between(xA, yA - yA_err, yA + yA_err, alpha=0.2, color='blue', label=band_label)
    for xi, yi, lbl in zip(xA, yA, flitz_labels):
        if lbl and lbl != "nan":
            axA.text(xi, yi + 0.03, lbl, ha='center', fontsize=18)
    axA.axhline(0.5, color='red', linestyle='--')
    if len(xA) > 0:
        axA.text(xA.max() + 0.05, 0.48, '50% accuracy threshold', color='red', fontsize=15, ha='right', va='top')
    axA.set_xlabel("Temperature")
    axA.set_ylabel("Accuracy")
    if band == "sem":
        axA.set_ylim(0.3, 1.05)                       # unchanged from the original
    else:
        lo = float(np.nanmin(yA - yA_err)) - 0.05     # the Bernoulli SD needs room
        hi = float(np.nanmax(yA + yA_err)) + 0.05
        axA.set_ylim(min(0.3, lo), max(1.05, hi))
    axA.legend(fontsize=14, loc='lower left')

    # ===================== (B) Temperature vs Average rating =====================
    sns.lineplot(ax=axB, x=xB, y=yB, marker='o', label='Average rating')
    axB.fill_between(xB, yB - yB_err, yB + yB_err, alpha=0.2, color='blue', label=band_b_label)
    axB.set_xlabel("Temperature")
    axB.set_ylabel("Average rating")
    axB.set_xlim(0.0, 2.0)
    axB.legend(fontsize=14, loc='upper right')

    # ===================== (C) Consecutive temperature distance =====================
    sns.lineplot(ax=axC, x=xC, y=yC, marker='o')
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
    add_panel_label_outside(axA, "(A)", fig=fig)
    add_panel_label_outside(axB, "(B)", fig=fig)
    add_panel_label_outside(axC, "(C)", fig=fig)
    add_panel_label_outside(axD, "(D)", fig=fig)

    out_png = out_png or f"fig3_phase_transitions_A_{band}.png"
    print(f"  (A) band = {band_label}, (B) band = {band_b_label}")
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_png}")

    # also save the panel-(A) spread numbers next to the figure
    spread.to_csv(RESULTS_DIR / "accuracy_spread_by_temp.csv", index=False)
    return spread


if __name__ == "__main__":
    # (A) SE, (B) untouched original SD
    s = make_figure(band="sem", band_b="sd")
    # (A) SD, (B) untouched original SD  -- the literal "same as (B)" reading
    make_figure(band="sd", band_b="sd")
    # "only SE": both (A) and (B) show the standard error of the mean
    make_figure(band="sem", band_b="sem",
                out_png="fig3_phase_transitions_all_SE.png")
    print()
    print(s.round(4).to_string(index=False))
