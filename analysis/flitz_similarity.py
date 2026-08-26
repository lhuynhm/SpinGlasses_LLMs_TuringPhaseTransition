# -*- coding: utf-8 -*-
"""
Flitz survey (2026) step 3 of 3 — human-human similarity on the new flitzes.

Same published method as the paper's human-human analysis: pairwise TF-IDF
cosine with a FRESH vectorizer per pair (IDF over the 2-document pair). Produces
a heatmap + the mean within-corpus similarity with a bootstrap CI, and prints the
original 33-flitz human baseline (from Data for Figures/human_written_flitzes.xlsx)
alongside it for direct comparison.

Note on N: this new corpus is small (however many uploads had usable text), so
its similarity mean is an estimate with a wide CI — report as descriptive.

Only real flitzes enter the cosine: uploads that carried no flitz text (the blank
"No Flitz" pages — 3 of the 10 responses) are dropped by flitz_extract.py and
never reach `flitz_survey2026_texts.xlsx`; load_texts() re-checks for empty rows.
The matrix diagonal is self-similarity and is 1 by definition; all reported means
are over the off-diagonal pairs only.

Outputs
    results/flitz_similarity_matrix.csv - full pairwise cosine matrix
    results/flitz_similarity.csv        - mean + bootstrap CI, vs original baseline
    results/flitz_similarity_heatmap.png

Run (after flitz_extract.py):
    .venv/bin/python analysis/flitz_similarity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import DATA_DIR, HUMAN_FLITZ_FILE, DEFAULT_RESULTS, ensure_dir, apply_style

TEXTS_XLSX = DATA_DIR / "flitz_survey2026_texts.xlsx"


def cosine_pair(a: str, b: str) -> float:
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([a, b])
    return 0.0 if tfidf.shape[1] == 0 else float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])


def full_matrix(texts):
    """Symmetric pairwise-cosine matrix with a self-similarity diagonal of 1.

    Only the upper triangle is computed (the metric is symmetric); the diagonal
    is set to 1 rather than left at the allocation value, since a document's
    cosine with itself is 1 by definition. Every consumer of the diagonal must
    therefore exclude it explicitly when averaging over *pairs*.
    """
    n = len(texts)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = cosine_pair(texts[i], texts[j])
    return M


def mean_offdiag(M):
    iu, ju = np.triu_indices(M.shape[0], k=1)
    return float(M[iu, ju].mean())


def bootstrap_ci(M, n_boot, rng):
    n = M.shape[0]
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        vals = [M[draw[i], draw[j]] for i in range(n) for j in range(i + 1, n)
                if draw[i] != draw[j]]
        means[b] = np.mean(vals) if vals else np.nan
    means = means[~np.isnan(means)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_texts(path, min_chars: int = 1):
    """Flitz texts, one per row, with non-flitzes dropped.

    `flitz_extract.py` already writes only the uploads that carried usable text
    (the blank "No Flitz" pages never reach this file), but the filter is
    repeated here so an empty or whitespace-only row can never enter the cosine.
    """
    col = pd.read_excel(path, header=None).iloc[:, 0].dropna().astype(str).str.strip()
    kept = col[col.str.len() >= min_chars]
    return kept.tolist(), int(len(col) - len(kept))


def count_blank_uploads(results_dir: Path) -> int:
    """How many uploads flitz_extract.py rejected as carrying no flitz text."""
    path = results_dir / "flitz_extracted.csv"
    if not path.exists():
        return 0
    ext = pd.read_csv(path)
    return int((~ext["has_text"].astype(bool)).sum()) if "has_text" in ext.columns else 0


def load_respondent_ids(results_dir: Path, n: int):
    """Source respondent id per kept flitz, in the order flitz_extract wrote them.

    Used only for provenance in the CSVs — the flitzes are *displayed* as 1..n
    (see `display_labels`). flitz_extract.py emits rows sorted by respondent_id,
    so 1..n runs in submission order.
    """
    path = results_dir / "flitz_extracted.csv"
    if not path.exists():
        return [None] * n
    ext = pd.read_csv(path)
    ext = ext[ext["has_text"]] if "has_text" in ext.columns else ext
    if len(ext) != n or "respondent_id" not in ext.columns:
        return [None] * n
    return [f"r{int(r)}" if pd.notna(r) else None for r in ext["respondent_id"]]


def display_labels(n: int):
    """Heatmap/matrix labels: the flitzes numbered 1..n, in submission order.

    The respondent ids they came from are non-contiguous (the blank uploads are
    dropped), so they are kept out of the figure and recorded in the CSVs instead.
    """
    return [str(i) for i in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_xlsx", default=str(TEXTS_XLSX))
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not Path(args.texts_xlsx).exists():
        raise SystemExit(f"{args.texts_xlsx} not found — run flitz_extract.py first.")

    rng = np.random.default_rng(args.seed)
    out_dir = ensure_dir(Path(args.out_dir))

    new_texts, dropped = load_texts(args.texts_xlsx)
    if dropped:
        print(f"[flitz] dropped {dropped} empty row(s) from {Path(args.texts_xlsx).name}.")
    n_blank = count_blank_uploads(out_dir)
    if n_blank:
        print(f"[flitz] {n_blank} upload(s) carried no flitz text and were excluded "
              f"upstream by flitz_extract.py; cosine uses the {len(new_texts)} real flitzes.")
    if len(new_texts) < 2:
        raise SystemExit(f"Only {len(new_texts)} usable flitz(es) — need >= 2 for pairwise similarity.")
    M = full_matrix(new_texts)
    new_mean = mean_offdiag(M)
    new_lo, new_hi = bootstrap_ci(M, args.n_boot, rng)

    # original 33-flitz human baseline for comparison
    base_texts, _ = load_texts(HUMAN_FLITZ_FILE)
    B = full_matrix(base_texts)
    base_mean = mean_offdiag(B)
    base_lo, base_hi = bootstrap_ci(B, args.n_boot, rng)

    labels = display_labels(len(new_texts))
    respondents = load_respondent_ids(out_dir, len(new_texts))
    mat = pd.DataFrame(M, index=labels, columns=labels)
    mat.index.name = "flitz"
    mat.to_csv(out_dir / "flitz_similarity_matrix.csv")
    # each flitz's mean similarity to the others (self-similarity excluded)
    per = (M.sum(axis=1) - np.diag(M)) / (len(new_texts) - 1)
    pd.DataFrame({"flitz": labels, "respondent_id": respondents,
                  "mean_cosine_to_others": per}).to_csv(
        out_dir / "flitz_similarity_per_flitz.csv", index=False)
    summary = pd.DataFrame([
        {"corpus": "flitz_survey_2026", "n_flitzes": len(new_texts),
         "mean_cosine": new_mean, "ci_low": new_lo, "ci_high": new_hi},
        {"corpus": "original_human_33", "n_flitzes": len(base_texts),
         "mean_cosine": base_mean, "ci_low": base_lo, "ci_high": base_hi},
    ])
    summary.to_csv(out_dir / "flitz_similarity.csv", index=False)

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title(f"New-flitz pairwise cosine (n={len(new_texts)}, "
                 f"off-diagonal mean={new_mean:.2f})")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    # small corpus: print the value in each cell (diagonal = self-similarity = 1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j] - 0.5) > 0.32 else "black")
    ax.grid(False)
    ax.set_xlabel("Flitz"); ax.set_ylabel("Flitz")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_dir / "flitz_similarity_heatmap.png", bbox_inches="tight")

    print(summary.to_string(index=False))
    print(f"[flitz] wrote flitz_similarity.csv / _matrix.csv / _heatmap.png to {out_dir}")


if __name__ == "__main__":
    main()
