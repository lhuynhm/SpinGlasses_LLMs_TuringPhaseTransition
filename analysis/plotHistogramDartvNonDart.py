# -*- coding: utf-8 -*-
"""
Score distribution: Dartmouth-affiliated vs. non-Dartmouth respondents.

The 2026 re-open (wave 2) added a free-text question, "What is your most recent
educational institution?", which is absent in wave 1 — so this split is a
wave-2-only analysis. Of the 20 wave-2 respondents, 11 named Dartmouth and 9
named somewhere else.

A respondent's score is the number of the 18 flitzes they classified correctly
(Human vs. AI). It is recomputed from the trial-level judgments rather than read
off the form's graded `Total score` column; the two agree exactly for all 20
wave-2 respondents.

Outputs
    results/dart_vs_nondart_scores.csv       - one row per respondent
    results/dart_vs_nondart_summary.csv      - group summary + Mann-Whitney U
    results/dart_vs_nondart_histogram.png    - two panels, left = Dartmouth

Run:
    ../.venv/bin/python plotHistogramDartvNonDart.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common import (
    DEFAULT_SURVEY, DEFAULT_SHEET, DEFAULT_RESULTS, DEFAULT_WAVE_CUTOFF,
    build_tidy_df, load_survey_frame, find_institution_col, is_dartmouth,
    ensure_dir, apply_style,
)

N_ITEMS = 18  # flitzes shown to every respondent

DART_COLOR = "#00693e"      # Dartmouth green
NONDART_COLOR = "#4c72b0"


def build_scores(survey_path, sheet_name, wave_cutoff, wave: int) -> pd.DataFrame:
    """One row per respondent: score, items answered, institution, group."""
    tidy = build_tidy_df(survey_path, sheet_name, wave_cutoff)
    sub = tidy[tidy["wave"] == wave]
    if sub.empty:
        raise SystemExit(f"No wave-{wave} responses in {survey_path}.")

    scores = sub.groupby("respondent_id").agg(
        score=("correct", "sum"),
        answered=("correct", "size"),
    ).reset_index()

    raw = load_survey_frame(survey_path, sheet_name).reset_index(drop=True)
    inst_col = find_institution_col(raw)
    if inst_col is None:
        raise SystemExit("No 'most recent educational institution' column found.")
    scores["institution"] = raw.loc[scores["respondent_id"], inst_col].values
    scores["is_dartmouth"] = scores["institution"].map(is_dartmouth)
    scores["group"] = scores["is_dartmouth"].map(
        {True: "Dartmouth", False: "Non-Dartmouth"}
    )
    return scores


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in scores.groupby("group"):
        rows.append({
            "group": group,
            "n": len(g),
            "mean": g["score"].mean(),
            "sd": g["score"].std(ddof=1),
            "median": g["score"].median(),
            "min": g["score"].min(),
            "max": g["score"].max(),
            "mean_accuracy": g["score"].mean() / N_ITEMS,
        })
    order = {"Dartmouth": 0, "Non-Dartmouth": 1}
    out = (pd.DataFrame(rows)
           .sort_values("group", key=lambda s: s.map(order))
           .reset_index(drop=True))

    dart = scores.loc[scores["is_dartmouth"] == True, "score"]        # noqa: E712
    other = scores.loc[scores["is_dartmouth"] == False, "score"]      # noqa: E712
    if len(dart) and len(other):
        u = mannwhitneyu(dart, other, alternative="two-sided")
        out["mw_u"] = u.statistic
        out["mw_p"] = u.pvalue
    return out


def make_figure(scores: pd.DataFrame, out_path: Path, wave: int) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    lo = int(scores["score"].min())
    hi = int(scores["score"].max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    panels = [
        ("Dartmouth", DART_COLOR),
        ("Non-Dartmouth", NONDART_COLOR),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, (group, color) in zip(axes, panels):
        g = scores[scores["group"] == group]["score"]
        ax.hist(g, bins=bins, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(g.mean(), ls="--", color="red", lw=1.5,
                   label=f"mean = {g.mean():.1f}")
        ax.axvline(N_ITEMS / 2, ls=":", color="gray", lw=1.5,
                   label=f"chance ({N_ITEMS // 2})")
        ax.set_title(f"{group} (n={len(g)})")
        ax.set_xlabel(f"Score (correct out of {N_ITEMS})")
        ax.set_xticks(np.arange(lo, hi + 1))
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Number of respondents")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.suptitle(
        f"Detection score distribution by institution — 2026 responses (wave {wave})"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", default=str(DEFAULT_SURVEY))
    ap.add_argument("--sheet_name", default=DEFAULT_SHEET)
    ap.add_argument("--wave_cutoff", default=DEFAULT_WAVE_CUTOFF)
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--wave", type=int, default=2,
                    help="Wave to split (the institution question is wave-2 only).")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    scores = build_scores(args.input_xlsx, args.sheet_name, args.wave_cutoff, args.wave)

    unknown = scores["is_dartmouth"].isna().sum()
    if unknown:
        print(f"[DvND] {unknown} wave-{args.wave} respondent(s) left the institution "
              f"question blank; excluded from the panels.")
    scores = scores.dropna(subset=["is_dartmouth"])

    scores_csv = out_dir / f"dart_vs_nondart_scores{args.suffix}.csv"
    scores.to_csv(scores_csv, index=False)

    summary = summarize(scores)
    summary_csv = out_dir / f"dart_vs_nondart_summary{args.suffix}.csv"
    summary.to_csv(summary_csv, index=False)

    fig_path = out_dir / f"dart_vs_nondart_histogram{args.suffix}.png"
    make_figure(scores, fig_path, args.wave)

    print(f"[DvND] wrote {scores_csv}")
    print(f"[DvND] wrote {summary_csv}")
    print(f"[DvND] wrote {fig_path}")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
