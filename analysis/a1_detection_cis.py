# -*- coding: utf-8 -*-
"""
A1 - Uncertainty on the detection curve.

For each AI temperature (and per wave + pooled): number of judgments, number of
correct detections, accuracy, Wilson 95% CI, exact binomial test vs chance(50% accuracy),
and Holm-corrected p-values across the 9 surveyed temperatures.

Outputs
    results/a1_detection_cis.csv   - one row per (wave, temperature)
    results/a1_detection_curve.png - accuracy vs temperature with Wilson CIs.
        With both collection waves present, this is TWO side-by-side panels
        titled "2025 results" (wave 1) and "2026 results" (wave 2); the waves are
        never drawn as two series in one axes. Only the chance line is named in
        the legend ("50% accuracy").

Run:
    .venv/bin/python analysis/a1_detection_cis.py
Once wave-2 data is added, this produces wave1 / wave2 / pooled automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

from common import (
    DEFAULT_SURVEY, DEFAULT_SHEET, DEFAULT_RESULTS, DEFAULT_WAVE_CUTOFF,
    build_tidy_df, ensure_dir, apply_style,
)


# Waves are collection years; the figure is labelled by year, not by wave number.
WAVE_YEAR_LABEL = {1: "2025 results", 2: "2026 results"}
WAVE_COLOR = {"wave1": "C0", "wave2": "C1", "pooled": "C0"}


def _year_label(w: int) -> str:
    return WAVE_YEAR_LABEL.get(int(w), f"wave {int(w)} results")


def detection_stats(ai_df: pd.DataFrame, wave_label: str) -> pd.DataFrame:
    rows = []
    for temp, g in ai_df.groupby("temperature"):
        n = len(g)
        successes = int(g["correct"].sum())
        acc = successes / n if n else np.nan
        lo, hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
        p = binomtest(successes, n, p=0.5, alternative="two-sided").pvalue
        flitz_num = int(g["flitz_number"].iloc[0])
        rows.append({
            "wave": wave_label, "temperature": temp, "flitz_number": flitz_num,
            "n": n, "successes": successes, "accuracy": acc,
            "ci_low": lo, "ci_high": hi, "p_exact": p,
        })
    out = pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)
    if len(out):
        out["p_holm"] = multipletests(out["p_exact"], method="holm")[1]
        out["sig_holm_05"] = out["p_holm"] < 0.05
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", default=str(DEFAULT_SURVEY))
    ap.add_argument("--sheet_name", default=DEFAULT_SHEET)
    ap.add_argument("--wave_cutoff", default=DEFAULT_WAVE_CUTOFF)
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--wave", default="all",
                    help="Which wave(s) to plot: 'all' (pooled + each wave), "
                         "'pooled', or a specific wave number e.g. '2' for the "
                         "2026 responses only.")
    ap.add_argument("--suffix", default="",
                    help="Suffix appended to output filenames, e.g. '_wave2'.")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    tidy = build_tidy_df(args.input_xlsx, args.sheet_name, args.wave_cutoff)
    tidy.to_csv(out_dir / f"tidy_df{args.suffix}.csv", index=False)

    ai = tidy[tidy["true_author"] == "AI"].dropna(subset=["temperature"])

    # Always compute the full CSV (pooled + every wave present).
    frames = [detection_stats(ai, "pooled")]
    for w in sorted(ai["wave"].unique()):
        sub = ai[ai["wave"] == w]
        if len(sub):
            frames.append(detection_stats(sub, f"wave{int(w)}"))
    result = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / f"a1_detection_cis{args.suffix}.csv"
    result.to_csv(out_csv, index=False)

    # Decide which curve(s) to draw. With both waves present, `--wave all` draws
    # them as two side-by-side panels ("2025 results" / "2026 results") rather
    # than as two series in one axes.
    waves_present = sorted({int(w) for w in ai["wave"].unique()})
    suptitle = "Detection accuracy with Wilson 95% CIs"
    if args.wave == "all":
        panels = ([("pooled", None)] if len(waves_present) <= 1
                  else [(f"wave{w}", _year_label(w)) for w in waves_present])
    elif args.wave == "pooled":
        panels = [("pooled", None)]
        suptitle = "Detection accuracy (pooled) with Wilson 95% CIs"
    else:
        w = int(args.wave)
        n_resp = ai[ai["wave"] == w]["respondent_id"].nunique()
        panels = [(f"wave{w}", f"{_year_label(w)} (n={n_resp})")]

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (wl, panel_title) in zip(axes, panels):
        d = result[result["wave"] == wl].sort_values("temperature")
        if not d.empty:
            yerr = np.vstack([d["accuracy"] - d["ci_low"], d["ci_high"] - d["accuracy"]])
            ax.errorbar(d["temperature"], d["accuracy"], yerr=yerr, marker="o",
                        capsize=3, color=WAVE_COLOR.get(wl, "C0"), lw=1.5)
        ax.axhline(0.5, ls="--", color="red", lw=1.5, label="50% accuracy")
        ax.set_xlabel("Generation temperature")
        ax.set_ylim(0, 1)
        if panel_title:
            ax.set_title(panel_title)
        ax.legend(frameon=False, loc="lower left")
    axes[0].set_ylabel("Detection accuracy")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig_path = out_dir / f"a1_detection_curve{args.suffix}.png"
    fig.savefig(fig_path, bbox_inches="tight")

    print(f"[A1] wrote {out_csv}")
    print(f"[A1] wrote {fig_path}")
    shown = "pooled" if args.wave == "all" else panels[0][0]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(result[result["wave"] == shown].to_string(index=False))


if __name__ == "__main__":
    main()
