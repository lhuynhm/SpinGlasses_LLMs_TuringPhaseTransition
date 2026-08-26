# -*- coding: utf-8 -*-
"""
A3 - Homogenization quantified on the FULL sample.

Within-AI pairwise TF-IDF cosine similarity per temperature, against TWO
within-human baselines: the original 33 human flitzes (528 pairs) and the 2026
flitz-survey corpus (7 flitzes, 21 pairs). Reuses the repo's published method
exactly: a FRESH TfidfVectorizer fit per pair (IDF over the 2-document pair), so
the numbers are comparable with Figs. 1/S2/S3.

Design decision (per the roadmap): within-AI pairs split into
  * cross-profile pairs  - two generations from DIFFERENT virtual students
  * same-profile pairs   - two generations from the SAME virtual student
The PRIMARY homogenization comparison is CROSS-PROFILE, the like-for-like
analogue of the human baseline (33 distinct authors, one flitz each). Same-profile
and pooled numbers go to the CSV for the SI.

Matched-size bootstrap: each iteration draws one flitz per profile (33 flitzes,
all cross-profile) and averages the 528 pairwise cosines -> CI in one loop.

CAVEAT for the caption: near-zero cosine at high T reflects vocabulary
disjointness from incoherence, NOT healthy creative diversity.

The per-temperature 330x330 cosine matrices are cached under
results/a3_cache/ (npz) so reruns are instant. First full run ~8 min.

Outputs
    results/a3_homogenization.csv          - per-temperature stats + CIs
    results/a3_homogenization.png          - within-AI vs the two within-human
                                             baselines, by temperature
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (
    DATA_DIR, HUMAN_FLITZ_FILE, AI_SAMPLES_DIR, DEFAULT_RESULTS, SURVEYED_TEMPS,
    ensure_dir, apply_style,
)

# The 2026 flitz-survey corpus (written by flitz_extract.py). Same human-written
# genre as HUMAN_FLITZ_FILE, collected a year later; used as a second, independent
# within-human baseline. Optional: if absent, that line is simply omitted.
NEW_HUMAN_FLITZ_FILE = DATA_DIR / "flitz_survey2026_texts.xlsx"

AI_PATTERN = re.compile(r"temp_(\d+\.\d{2})_sample_(\d+)\.xlsx")


def cosine_pair(a: str, b: str) -> float:
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([a, b])
    if tfidf.shape[1] == 0:
        return 0.0
    return float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])


def full_matrix(texts) -> np.ndarray:
    # NB: the diagonal is left at 0 rather than the true self-similarity of 1.
    # That is safe *here* because every consumer reads off-diagonal pairs only
    # (triu k=1), and the cached a3_cache/*.npz matrices were written this way.
    # Do not average a row of this matrix directly. (flitz_similarity.py, which
    # does display its matrix, sets the diagonal to 1.)
    n = len(texts)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_pair(texts[i], texts[j])
            M[i, j] = M[j, i] = s
    return M


def load_human_texts(path=HUMAN_FLITZ_FILE):
    df = pd.read_excel(path, header=None)  # no header row!
    col = df.iloc[:, 0].dropna().astype(str).str.strip()
    return col[col.str.len() > 0].tolist()


def load_ai_for_temp(temp_str: str):
    """Return texts, profile ids (Prompt Index), sample numbers for one temp."""
    texts, profiles, samples = [], [], []
    for fname in sorted(AI_SAMPLES_DIR.iterdir()):
        m = AI_PATTERN.match(fname.name)
        if not m or m.group(1) != temp_str:
            continue
        sample_num = int(m.group(2))
        df = pd.read_excel(fname)
        col = "Flitz Output" if "Flitz Output" in df.columns else df.columns[-1]
        pcol = "Prompt Index" if "Prompt Index" in df.columns else None
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            texts.append(str(val).strip())
            profiles.append(int(df[pcol].iloc[idx]) if pcol else idx)
            samples.append(sample_num)
    return texts, np.array(profiles), np.array(samples)


def cached_matrix(temp_str: str, cache_dir: Path):
    cache_dir = ensure_dir(cache_dir)
    fp = cache_dir / f"cos_temp_{temp_str}.npz"
    if fp.exists():
        z = np.load(fp)
        return z["M"], z["profiles"], z["samples"]
    texts, profiles, samples = load_ai_for_temp(temp_str)
    M = full_matrix(texts)
    np.savez_compressed(fp, M=M, profiles=profiles, samples=samples)
    return M, profiles, samples


def strata_means(M, profiles):
    """Mean cosine over cross-profile pairs, same-profile pairs, and pooled."""
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    same = profiles[iu] == profiles[ju]
    vals = M[iu, ju]
    cross = vals[~same]
    within = vals[same]
    return {
        "cross_profile_mean": float(cross.mean()) if len(cross) else np.nan,
        "same_profile_mean": float(within.mean()) if len(within) else np.nan,
        "pooled_mean": float(vals.mean()) if len(vals) else np.nan,
        "n_cross_pairs": int(len(cross)), "n_same_pairs": int(len(within)),
    }


def matched_bootstrap(M, profiles, samples, n_boot, rng):
    """One flitz per profile per iteration -> mean of 528 cross-profile pairs."""
    prof_to_idx = {}
    for i, p in enumerate(profiles):
        prof_to_idx.setdefault(int(p), []).append(i)
    prof_ids = sorted(prof_to_idx)
    means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = [prof_to_idx[p][rng.integers(len(prof_to_idx[p]))] for p in prof_ids]
        sub = M[np.ix_(chosen, chosen)]
        iu, ju = np.triu_indices(len(chosen), k=1)
        means[b] = sub[iu, ju].mean()
    return means


def human_baseline(texts, n_boot, rng):
    M = full_matrix(texts)
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    point = float(M[iu, ju].mean())
    # author-level bootstrap: resample authors with replacement, average over
    # pairs of DISTINCT drawn authors (skip self-pairs from duplicates).
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        vals = [M[draw[i], draw[j]] for i in range(n) for j in range(i + 1, n)
                if draw[i] != draw[j]]
        means[b] = np.mean(vals) if vals else np.nan
    return point, means, M


def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temps", nargs="*", default=None,
                    help="Restrict to these temp strings, e.g. 0.00 0.25 ... "
                         "(default: all 17 in the corpus)")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    cache_dir = out_dir / "a3_cache"
    rng = np.random.default_rng(args.seed)

    # discover temps present
    all_temps = sorted({AI_PATTERN.match(f.name).group(1)
                        for f in AI_SAMPLES_DIR.iterdir()
                        if AI_PATTERN.match(f.name)}, key=float)
    temps = args.temps if args.temps else all_temps

    print("[A3] human within-baseline (33 flitzes) ...")
    h_point, h_boot, _ = human_baseline(load_human_texts(), args.n_boot, rng)
    h_lo, h_hi = ci(h_boot)

    # Second within-human baseline: the 2026 survey flitzes. Drawn on its own
    # rng stream (seed+1) so adding it leaves every pre-existing number in this
    # script bit-identical.
    n2 = h2_point = h2_lo = h2_hi = None
    if NEW_HUMAN_FLITZ_FILE.exists():
        new_texts = load_human_texts(NEW_HUMAN_FLITZ_FILE)
        if len(new_texts) >= 2:
            n2 = len(new_texts)
            print(f"[A3] human within-baseline ({n2} new flitzes) ...")
            h2_point, h2_boot, _ = human_baseline(
                new_texts, args.n_boot, np.random.default_rng(args.seed + 1))
            h2_lo, h2_hi = ci(h2_boot)
    if h2_point is None:
        print(f"[A3] {NEW_HUMAN_FLITZ_FILE.name} unusable/absent - "
              "new-human baseline omitted.")

    rows = []
    for t in temps:
        print(f"[A3] temperature {t} ...")
        M, profiles, samples = cached_matrix(t, cache_dir)
        st = strata_means(M, profiles)
        boot = matched_bootstrap(M, profiles, samples, args.n_boot, rng)
        lo, hi = ci(boot)
        rows.append({
            "temperature": float(t),
            "ai_cross_profile_mean": st["cross_profile_mean"],
            "ai_cross_ci_low": lo, "ai_cross_ci_high": hi,
            "ai_same_profile_mean": st["same_profile_mean"],
            "ai_pooled_mean": st["pooled_mean"],
            "n_cross_pairs": st["n_cross_pairs"],
            "n_same_pairs": st["n_same_pairs"],
            "human_baseline_mean": h_point,
            "human_ci_low": h_lo, "human_ci_high": h_hi,
            "human2026_baseline_mean": h2_point,
            "human2026_ci_low": h2_lo, "human2026_ci_high": h2_hi,
            "n_human2026_flitzes": n2,
        })
    result = pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)
    out_csv = out_dir / "a3_homogenization.csv"
    result.to_csv(out_csv, index=False)

    # --- Figure -------------------------------------------------------------
    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    yerr = np.vstack([
        result["ai_cross_profile_mean"] - result["ai_cross_ci_low"],
        result["ai_cross_ci_high"] - result["ai_cross_profile_mean"],
    ])
    ax.errorbar(result["temperature"], result["ai_cross_profile_mean"], yerr=yerr,
                marker="o", capsize=3, lw=1.5, color="#d1495b",
                label="within-AI (cross-profile)")
    ax.axhline(h_point, color="#2e86ab", lw=1.5,
               label="within-human baseline (n=33)")
    ax.axhspan(h_lo, h_hi, color="#2e86ab", alpha=0.15)
    if h2_point is not None:
        # dashed as well as differently coloured: the two baselines stay
        # distinguishable for colour-vision-deficient readers and in print.
        ax.axhline(h2_point, color="#c77b1f", lw=1.5, ls="--",
                   label=f"within-human baseline, 2026 flitzes (n={n2})")
        ax.axhspan(h2_lo, h2_hi, color="#c77b1f", alpha=0.12)
    ax.set_xlabel("Generation temperature")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Homogenization: within-AI vs within-human similarity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "a3_homogenization.png", bbox_inches="tight")

    print(f"[A3] human baseline (n=33) mean={h_point:.3f} CI=({h_lo:.3f},{h_hi:.3f})")
    if h2_point is not None:
        print(f"[A3] human baseline (2026, n={n2}) mean={h2_point:.3f} "
              f"CI=({h2_lo:.3f},{h2_hi:.3f})")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(result.to_string(index=False))
    print(f"[A3] wrote {out_csv} and a3_homogenization.png")


if __name__ == "__main__":
    main()
