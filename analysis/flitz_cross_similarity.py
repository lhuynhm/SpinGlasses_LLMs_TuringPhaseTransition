# -*- coding: utf-8 -*-
"""
Cross-corpus similarity: the 2026 flitz survey (7 flitzes) vs the original 2025
human corpus (33 flitzes).

Same published method as every other cosine in the repo, reused verbatim from
`flitz_similarity.cosine_pair`: pairwise TF-IDF cosine with a FRESH vectorizer
per pair (IDF over the 2-document pair). Here the matrix is RECTANGULAR - rows
are the 7 new (2026) flitzes, columns the 33 original (2025) flitzes - so every
one of the 7*33 = 231 cells is a genuine cross-corpus pair and there is no
diagonal to exclude.

Reported for each corpus pairing: mean cosine, SD over pairs, and the same
bootstrap CI style used in flitz_similarity.py (resampling authors, not pairs).
The two within-corpus numbers (2025 x 2025, 2026 x 2026) are recomputed here so
the three lines of the summary are directly comparable.

Outputs
    results/flitz_cross_similarity_matrix.csv  - the full 7x33 cosine matrix
    results/flitz_cross_similarity_margins.csv - per-flitz means on both margins
    results/flitz_cross_similarity.csv         - mean/SD/CI: cross + both within
    results/flitz_cross_similarity_heatmap.png

Run (after flitz_extract.py):
    .venv/bin/python analysis/flitz_cross_similarity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import HUMAN_FLITZ_FILE, DEFAULT_RESULTS, ensure_dir, apply_style
from flitz_similarity import (
    TEXTS_XLSX, cosine_pair, full_matrix, mean_offdiag, bootstrap_ci,
    load_texts, load_respondent_ids, display_labels,
)


def cross_matrix(rows, cols):
    """Rectangular pairwise-cosine matrix, one row per `rows` doc, one column per
    `cols` doc. No diagonal to mask: the two corpora are disjoint."""
    M = np.empty((len(rows), len(cols)))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            M[i, j] = cosine_pair(a, b)
    return M


def cross_bootstrap_ci(M, n_boot, rng):
    """CI on the cross-corpus mean, resampling the *authors* on both margins.

    Mirrors flitz_similarity.bootstrap_ci (which resamples the authors of one
    corpus); here rows and columns are independent corpora, so both are drawn
    with replacement. Every drawn cell is a valid cross pair, so - unlike the
    within-corpus case - nothing has to be discarded.
    """
    n, m = M.shape
    means = np.empty(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, n, size=n)
        ci = rng.integers(0, m, size=m)
        means[b] = M[np.ix_(ri, ci)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def offdiag_values(M):
    iu, ju = np.triu_indices(M.shape[0], k=1)
    return M[iu, ju]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_xlsx", default=str(TEXTS_XLSX),
                    help="2026 flitz-survey texts (from flitz_extract.py)")
    ap.add_argument("--baseline_xlsx", default=str(HUMAN_FLITZ_FILE),
                    help="original 2025 33-flitz human corpus")
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not Path(args.texts_xlsx).exists():
        raise SystemExit(f"{args.texts_xlsx} not found — run flitz_extract.py first.")

    rng = np.random.default_rng(args.seed)
    out_dir = ensure_dir(Path(args.out_dir))

    new_texts, dropped_new = load_texts(args.texts_xlsx)
    base_texts, dropped_base = load_texts(args.baseline_xlsx)
    for name, dropped in [(Path(args.texts_xlsx).name, dropped_new),
                          (Path(args.baseline_xlsx).name, dropped_base)]:
        if dropped:
            print(f"[flitz] dropped {dropped} empty row(s) from {name}.")
    if not new_texts or not base_texts:
        raise SystemExit("Both corpora need at least one flitz for the cross matrix.")

    C = cross_matrix(new_texts, base_texts)
    cross_mean = float(C.mean())
    cross_sd = float(C.std(ddof=1))
    cross_lo, cross_hi = cross_bootstrap_ci(C, args.n_boot, rng)

    # the two within-corpus references, on the same scale (off-diagonal pairs only)
    N = full_matrix(new_texts)
    B = full_matrix(base_texts)
    new_vals, base_vals = offdiag_values(N), offdiag_values(B)

    row_labels = display_labels(len(new_texts))     # 2026 flitzes, 1..7
    col_labels = display_labels(len(base_texts))    # 2025 flitzes, 1..33

    mat = pd.DataFrame(C, index=row_labels, columns=col_labels)
    mat.index.name = "flitz_2026"
    mat.columns.name = "flitz_2025"
    mat.to_csv(out_dir / "flitz_cross_similarity_matrix.csv")

    # marginal means: how similar each flitz is to the *other* corpus as a whole
    margins = pd.concat([
        pd.DataFrame({"corpus": "flitz_survey_2026", "flitz": row_labels,
                      "respondent_id": load_respondent_ids(out_dir, len(new_texts)),
                      "mean_cosine_to_other_corpus": C.mean(axis=1),
                      "sd_cosine_to_other_corpus": C.std(axis=1, ddof=1)}),
        pd.DataFrame({"corpus": "original_human_33_2025", "flitz": col_labels,
                      "respondent_id": None,
                      "mean_cosine_to_other_corpus": C.mean(axis=0),
                      "sd_cosine_to_other_corpus": C.std(axis=0, ddof=1)}),
    ], ignore_index=True)
    margins.to_csv(out_dir / "flitz_cross_similarity_margins.csv", index=False)

    summary = pd.DataFrame([
        {"comparison": "2026_x_2025 (cross)", "n_a": len(new_texts), "n_b": len(base_texts),
         "n_pairs": C.size, "mean_cosine": cross_mean, "sd_cosine": cross_sd,
         "ci_low": cross_lo, "ci_high": cross_hi},
        {"comparison": "2026_x_2026 (within)", "n_a": len(new_texts), "n_b": len(new_texts),
         "n_pairs": new_vals.size, "mean_cosine": mean_offdiag(N),
         "sd_cosine": float(new_vals.std(ddof=1)),
         **dict(zip(("ci_low", "ci_high"), bootstrap_ci(N, args.n_boot, rng)))},
        {"comparison": "2025_x_2025 (within)", "n_a": len(base_texts), "n_b": len(base_texts),
         "n_pairs": base_vals.size, "mean_cosine": mean_offdiag(B),
         "sd_cosine": float(base_vals.std(ddof=1)),
         **dict(zip(("ci_low", "ci_high"), bootstrap_ci(B, args.n_boot, rng)))},
    ])
    summary.to_csv(out_dir / "flitz_cross_similarity.csv", index=False)

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.46 * len(col_labels) + 2.2,
                                    0.55 * len(row_labels) + 1.8))
    im = ax.imshow(C, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"2026 flitzes x 2025 human flitzes — pairwise cosine "
                 f"({len(row_labels)}x{len(col_labels)} = {C.size} pairs, "
                 f"mean={cross_mean:.2f}, SD={cross_sd:.2f})")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{C[i, j]:.2f}"[1:], ha="center", va="center", fontsize=5,
                    color="white" if abs(C[i, j] - 0.5) > 0.32 else "black")
    ax.grid(False)
    ax.set_xlabel("2025 flitz (original 33-flitz human corpus)")
    ax.set_ylabel("2026 flitz")
    fig.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / "flitz_cross_similarity_heatmap.png", bbox_inches="tight")

    print(summary.to_string(index=False))
    print(f"[flitz] wrote flitz_cross_similarity.csv / _matrix.csv / _margins.csv / "
          f"_heatmap.png to {out_dir}")


if __name__ == "__main__":
    main()
