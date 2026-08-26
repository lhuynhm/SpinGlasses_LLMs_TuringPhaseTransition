# make_figure_2x2_human_vs_llm_judges.py
# -*- coding: utf-8 -*-
"""
2x2 figure: correctness vs. rating, split by the TRUE author of the flitz
(AI = orange, Human = blue), one panel per judge.

    top-left     Human survey respondents   (mean +- SE + logistic fit, as in the
                                             original panel (B))
    top-right    ChatGPT 5.6 Sol
    bottom-left  Claude Fable 5
    bottom-right Gemini 3.1 Pro

Style follows
`make_figures_consistency_phasetransitions_data.fig_AB_subplots_accuracy_dev_and_logistic_OUTSIDE`.

Every judge is now shown with the same layout: binned mean +- SE plus a dashed
logistic fit.  Each LLM was run 10 times over the full set of 18 flitzes, so the
model panels have a real per-bin spread; where a model is correct at every rating
the logistic MLE degenerates to the constant fit (a flat dashed line at 1).

Usage:
    python make_figure_2x2_human_vs_llm_judges.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_statistical_analysis import (
    detect_columns,
    build_tidy_df,
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
OUT_PNG = "figure_2x2_human_vs_llm_judges.png"
FIG_DPI = 300

# Same flitz -> author mapping used by run_statistical_analysis.build_tidy_df
HUMAN_FLITZ_NUMBERS = {1, 2, 6, 7, 8, 11, 12, 13, 16}

COLORS = {"AI": "orange", "Human": "blue"}
YLIM = (-0.08, 1.38)
XLIM = (0.5, 10.5)

# panel position -> judge; "Human" means the survey respondents
PANELS = [
    [("Human", "Human"), ("ChatGPT 5.6 Sol", "model")],
    [("Claude Fable 5", "model"), ("Gemini 3.1 Pro", "model")],
]
PANEL_LETTERS = [["(A)", "(B)"], ["(C)", "(D)"]]


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
    TuringTestTakenByAI.xlsx layout (no header row).  One block per judge:

        <model name>                       <- block header row
        'No.1' | verdict | rating | verdict | rating | ...   <- 18 flitz rows
        ...
        'No.18'| ...

    Each independent run of a model occupies two columns (verdict, rating),
    starting at column B, so run k lives in columns (2k-1, 2k) 0-indexed.
    Only the verdict 'Human' is written out; an EMPTY verdict cell means 'AI'.
    """
    raw = pd.read_excel(AI_JUDGE_XLSX, header=None)

    # Block header rows: first cell is a model name rather than 'No.<k>'.
    is_flitz_row = raw[0].astype(str).str.match(r'\s*No\.\s*\d+', na=False)
    header_rows = [i for i in raw.index if not is_flitz_row[i] and pd.notna(raw.at[i, 0])]

    rows = []
    for h, nxt in zip(header_rows, header_rows[1:] + [raw.index.max() + 1]):
        model = str(raw.at[h, 0]).strip()
        block = raw.loc[h + 1:nxt - 1]
        block = block[is_flitz_row.loc[block.index]]

        flitz_numbers = block[0].astype(str).str.extract(r'(\d+)')[0].astype(int)
        true_author = np.where(flitz_numbers.isin(HUMAN_FLITZ_NUMBERS), 'Human', 'AI')

        run = 0
        while True:                                   # walk the (verdict, rating) pairs
            verdict_col, rating_col = 1 + 2 * run, 2 + 2 * run
            if rating_col not in block.columns:
                break
            rating = pd.to_numeric(block[rating_col], errors="coerce")
            if rating.isna().all():                   # first empty run -> block is done
                break
            verdict = block[verdict_col].astype(str).str.strip()
            # blank cell == 'AI' (only 'Human' is spelled out in the sheet)
            predicted = np.where(verdict.str.lower().eq('human'), 'Human', 'AI')
            rows.append(pd.DataFrame({
                "model": model,
                "run": run + 1,
                "flitz_number": flitz_numbers.values,
                "true_author": true_author,
                "predicted_author": predicted,
                "correct": (predicted == true_author).astype(int),
                "rating": rating.values,
            }))
            run += 1
    return pd.concat(rows, ignore_index=True)


def draw_panel(ax, bins_stats, preds):
    """The original panel (B) layout: binned mean +- SE plus a dashed logistic fit."""
    for author in ["AI", "Human"]:
        s = preds[preds["Author"] == author]
        ax.plot(s["Rating"].to_numpy(), s["Predicted_Probability"].to_numpy(),
                linestyle='--', linewidth=4, color=COLORS[author],
                label=f"{author} (logistic fit)")
    for author in ["AI", "Human"]:
        s = bins_stats[bins_stats["author_type"] == author]
        ax.errorbar(s["rating_bin_mid"].astype(float).to_numpy(),
                    s["mean_correct"].to_numpy(),
                    yerr=s["sem_correct"].to_numpy(),
                    fmt='o', capsize=5, color=COLORS[author],
                    label=f"{author} (mean ± SE)")
    ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=11.5, columnspacing=1.0, handlelength=2.2,
              borderpad=0.3, labelspacing=0.3)


def main():
    tidy = load_survey_tidy()
    bins_stats = compute_binned_means_for_fig3(tidy)
    preds = compute_logistic_predictions(tidy, n_points=100)
    judges = load_ai_judges()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            name, kind = PANELS[r][c]
            if kind == "Human":
                draw_panel(ax, bins_stats, preds)
            else:
                sub = judges[judges["model"] == name]
                if sub.empty:
                    raise ValueError(f"No rows for model '{name}' in {AI_JUDGE_XLSX}")
                # Same estimators as the human panel; include_lowest keeps the
                # rating-1 verdicts, which the LLM judges use heavily.
                draw_panel(ax,
                           compute_binned_means_for_fig3(sub, include_lowest=True),
                           compute_logistic_predictions(sub, n_points=100))

            ax.set_title(name, fontsize=20, fontweight='bold', pad=12)
            ax.set_xlabel("Rating (1–10)")
            ax.set_ylabel("1=correct, 0=incorrect")
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_xticks(np.arange(2, 11, 2))

    fig.tight_layout()
    fig.canvas.draw()
    for r in range(2):
        for c in range(2):
            add_panel_label_outside(axes[r][c], PANEL_LETTERS[r][c], fig=fig)

    fig.savefig(OUT_PNG, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Wrote {OUT_PNG}")

    n_runs = judges.groupby("model")["run"].nunique()
    print(f"\nLLM judges, accuracy by true author "
          f"(9 flitzes x {n_runs.min()}-{n_runs.max()} runs each):")
    for model, sub in judges.groupby("model", sort=False):
        parts = []
        for author in ["AI", "Human"]:
            s = sub[sub["true_author"] == author]
            parts.append(f"{author}: {int(s['correct'].sum())}/{len(s)}")
        wrong = (sub[sub["correct"] == 0]
                 .groupby("flitz_number").size().to_dict())
        print(f"  {model:<16} " + "  ".join(parts)
              + (f"  misclassified {{flitz: n_runs}} = {wrong}" if wrong else ""))


if __name__ == "__main__":
    main()
