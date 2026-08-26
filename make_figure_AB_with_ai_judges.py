# make_figure_AB_with_ai_judges.py
# -*- coding: utf-8 -*-
"""
Reproduce the two-panel figure

    (A) |Accuracy - 0.50| vs. temperature
    (B) correctness vs. rating, split by true author (AI / Human)

in exactly the style of
`make_figures_consistency_phasetransitions_data.fig_AB_subplots_accuracy_dev_and_logistic_OUTSIDE`,
and additionally overlay panel (B) with one line per LLM that took the same
Turing test itself (Data for Figures/TuringTestTakenByAI.xlsx).

Each model answered every flitz exactly once, so no standard errors are drawn
for those lines (a single sample per flitz).

Usage:
    python make_figure_AB_with_ai_judges.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_statistical_analysis import (
    detect_columns,
    build_tidy_df,
    compute_accuracy_by_temp,
    compute_binned_means_for_fig3,
    compute_logistic_predictions,
)

# ---- GLOBAL STYLE (identical to make_figures_consistency_phasetransitions_data.py) ----
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 4,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.markersize': 10
})

DATA_DIR = "Data for Figures"
SURVEY_XLSX = f"{DATA_DIR}/04_12_ Distinguishing between AI vs. Human-written Flitzes (Responses).xlsx"
SURVEY_SHEET = "Form Responses 1"
AI_JUDGE_XLSX = f"{DATA_DIR}/TuringTestTakenByAI.xlsx"
OUT_PNG = "figure_AB_accuracydev_and_logistic_with_ai_judges.png"
FIG_DPI = 300

# Same flitz -> author mapping used by run_statistical_analysis.build_tidy_df
HUMAN_FLITZ_NUMBERS = {1, 2, 6, 7, 8, 11, 12, 13, 16}

# Same rating bins as compute_binned_means_for_fig3, so the model markers sit on
# the same x-grid as the survey markers.
RATING_BINS = np.arange(1, 11.5, 0.5)
RATING_LABELS = (RATING_BINS[:-1] + RATING_BINS[1:]) / 2

# The three models are correct almost everywhere, so their lines would sit on top
# of each other at y = 1. `dy` applies a small purely cosmetic vertical offset so
# all three remain visible; this is annotated in the panel.
MODEL_STYLE = {
    "Claude Fable 5":  dict(color="#2ca02c", linestyle="-",  marker="s", dy=+0.020),
    "ChatGPT 5.6 Sol": dict(color="#d62728", linestyle="-.", marker="^", dy=0.000),
    "Gemini 3.1 Pro":  dict(color="#8c564b", linestyle=":",  marker="D", dy=-0.020),
}


def add_panel_label_outside(ax, label, fig=None, pad=0.012, xpad=0.0, fontsize=25):
    """Place a bold panel label just ABOVE the axes (outside the plotting area)."""
    if fig is None:
        fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_position()
    fig.text(bbox.x0 + xpad, bbox.y1 + pad, label,
             fontsize=fontsize, fontweight='bold', ha='left', va='bottom')


def load_survey_tidy():
    df = pd.read_excel(SURVEY_XLSX, sheet_name=SURVEY_SHEET)
    pred_cols, rating_cols = detect_columns(df)
    return build_tidy_df(df, pred_cols, rating_cols)


def load_ai_judges():
    """
    TuringTestTakenByAI.xlsx layout (no header row):
        col 0            : 'No.1' ... 'No.18'
        cols 1,3,5       : the model's verdict ('Human' / 'AI')
        cols 2,4,6       : the model's rating (1-10)
        last row, cols 1,3,5 : model names
    Returns a tidy frame: model, flitz_number, true_author, predicted_author,
    correct, rating.
    """
    raw = pd.read_excel(AI_JUDGE_XLSX, header=None)

    name_row = raw.iloc[-1]
    data = raw.iloc[:-1].copy()
    flitz_numbers = data[0].astype(str).str.extract(r'(\d+)')[0].astype(int)
    true_author = np.where(flitz_numbers.isin(HUMAN_FLITZ_NUMBERS), 'Human', 'AI')

    rows = []
    for verdict_col, rating_col in ((1, 2), (3, 4), (5, 6)):
        model = str(name_row[verdict_col]).strip()
        predicted = data[verdict_col].astype(str).str.strip()
        rows.append(pd.DataFrame({
            "model": model,
            "flitz_number": flitz_numbers.values,
            "true_author": true_author,
            "predicted_author": predicted.values,
            "correct": (predicted.values == true_author).astype(int),
            "rating": pd.to_numeric(data[rating_col], errors="coerce").values,
        }))
    return pd.concat(rows, ignore_index=True)


def bin_model_accuracy(judges: pd.DataFrame):
    """
    Bin each model's correctness onto the same rating grid the survey uses.
    Note: pd.cut with these bins is right-closed and excludes the lower edge, so
    a rating of exactly 1 falls outside the grid -- the same treatment the survey
    responses get in compute_binned_means_for_fig3.
    """
    out = []
    for model, sub in judges.groupby("model", sort=False):
        sub = sub.dropna(subset=["rating"]).copy()
        sub["rating_bin"] = pd.cut(sub["rating"], bins=RATING_BINS, labels=RATING_LABELS)
        g = sub.groupby("rating_bin", observed=False)["correct"]
        tmp = pd.DataFrame({
            "model": model,
            "rating_bin_mid": g.mean().index.astype(float),
            "mean_correct": g.mean().values,
            "n": g.count().values,
        })
        out.append(tmp[tmp["n"] > 0])
    return pd.concat(out, ignore_index=True)


def main():
    tidy = load_survey_tidy()

    # ---------- (A) data ----------
    acc = compute_accuracy_by_temp(tidy).sort_values("temperature")
    xA = acc["temperature"].to_numpy()
    abs_dev = np.abs(acc["accuracy"].to_numpy() - 0.5)

    # ---------- (B) data ----------
    bins_stats = compute_binned_means_for_fig3(tidy)
    preds = compute_logistic_predictions(tidy, n_points=100)
    model_stats = bin_model_accuracy(load_ai_judges())
    colors = {"AI": "orange", "Human": "blue"}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 6))

    # ---------- (A) ----------
    axA.plot(xA, abs_dev, marker='o', linestyle='-', color='purple', linewidth=4)
    axA.axhline(0, linestyle='--', color='red')
    for i in range(len(xA)):
        y_offset, va = (-0.03, 'top') if abs_dev[i] > 0.35 else (0.02, 'bottom')
        axA.text(xA[i], abs_dev[i] + y_offset, f'{abs_dev[i]:.2f}',
                 ha='center', va=va, fontsize=15, color='black')
    axA.set_xlabel("Temperature")
    axA.set_ylabel("|Accuracy − 0.50|")
    axA.set_xticks(xA)
    axA.set_ylim(-0.02, 0.45)
    axA.text(xA.max() + 0.05, 0.01, '50% accuracy threshold',
             color='red', fontsize=15, ha='right', va='bottom')

    # ---------- (B) survey respondents ----------
    fit_handles, pt_handles = [], []
    for author in ["AI", "Human"]:
        s = preds[preds["Author"] == author]
        fit_handles.append(axB.plot(
            s["Rating"].to_numpy(),
            s["Predicted_Probability"].to_numpy(),
            linestyle='--', linewidth=4, color=colors[author],
            label=f"{author} (logistic fit)"
        )[0])
    for author in ["AI", "Human"]:
        s = bins_stats[bins_stats["author_type"] == author]
        pt_handles.append(axB.errorbar(
            s["rating_bin_mid"].astype(float).to_numpy(),
            s["mean_correct"].to_numpy(),
            yerr=s["sem_correct"].to_numpy(),
            fmt='o', capsize=5, color=colors[author],
            label=f"{author} (mean ± SE)"
        ))

    # ---------- (B) LLM judges (one sample per flitz -> no SE) ----------
    model_handles = []
    for model, style in MODEL_STYLE.items():
        s = model_stats[model_stats["model"] == model].sort_values("rating_bin_mid")
        if s.empty:
            continue
        style = dict(style)
        dy = style.pop("dy", 0.0)
        model_handles.append(axB.plot(
            s["rating_bin_mid"].to_numpy(),
            s["mean_correct"].to_numpy() + dy,
            linewidth=2.5, markersize=9,
            label=model, **style
        )[0])

    axB.set_xlabel("Rating (1–10)")
    axB.set_ylabel("1=correct, 0=incorrect")
    axB.set_ylim(0.1, 1.6)
    axB.text(0.5, 0.70, "LLM judges: n = 1 per flitz (no SE); lines offset vertically for visibility",
             transform=axB.transAxes, fontsize=11.5, color='0.35', ha='center', va='center')
    handles = fit_handles + pt_handles + model_handles
    axB.legend(handles, [h.get_label() for h in handles],
               title=None, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=13, columnspacing=1.0, handlelength=2.2,
               borderpad=0.3, labelspacing=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    add_panel_label_outside(axA, "(A)", fig=fig)
    add_panel_label_outside(axB, "(B)", fig=fig)

    fig.savefig(OUT_PNG, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Wrote {OUT_PNG}")

    # console summary of the three added lines
    judges = load_ai_judges()
    print("\nOverall accuracy of the LLM judges (18 flitzes each):")
    for model, sub in judges.groupby("model", sort=False):
        wrong = sub[sub["correct"] == 0]["flitz_number"].tolist()
        print(f"  {model:<16} {sub['correct'].mean():.3f}"
              f"  ({sub['correct'].sum()}/{len(sub)})"
              f"{'  misclassified flitz ' + str(wrong) if wrong else ''}")


if __name__ == "__main__":
    main()
