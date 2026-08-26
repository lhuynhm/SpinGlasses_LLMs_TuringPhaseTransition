# -*- coding: utf-8 -*-
"""
A2 - Trial-level craft-heuristic model (replaces the Fig. 3B aggregated fit).

Two deliberate changes vs the repo's compute_logistic_predictions():
  (i)  outcome is the VERDICT, P(judged "Human"), not `correct`; the craft
       heuristic is a claim about what makes a reader SAY "human".
  (ii) respondent clustering is modelled: GEE logistic with an exchangeable
       working correlation, cluster = respondent_id, giving cluster-robust SEs.
       (A full GLMM adds little here and needs extra tooling; cluster-robust
        GEE is the pragmatic, defensible choice.)

Model:      verdict_human ~ rating * C(true_author)     (separate slopes)
Restricted: same model on AI trials with T <= 1.0  +  all human trials
            (the "passing regime" — tests whether the heuristic operates even
             where detection is at chance).

Outputs
    results/a2_craft_coefs_full.csv        - ORs, CIs, robust p (full sample)
    results/a2_craft_coefs_restricted.csv  - same, passing-regime refit
    results/a2_craft_predictions.csv       - fitted P(human) + CI bands
    results/a2_craft_heuristic.png         - curves with CI bands (full + restricted)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

from common import (
    DEFAULT_SURVEY, DEFAULT_SHEET, DEFAULT_RESULTS, DEFAULT_WAVE_CUTOFF,
    build_tidy_df, ensure_dir, apply_style,
)

FORMULA = "verdict_human ~ rating * C(true_author, Treatment('AI'))"


def fit_gee(df: pd.DataFrame):
    df = df.dropna(subset=["rating", "verdict_human"]).copy()
    model = smf.gee(
        FORMULA, groups="respondent_id", data=df,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit()


def coef_table(res, label: str) -> pd.DataFrame:
    params = res.params
    bse = res.bse  # robust (GEE) SEs
    z = params / bse
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(z)))
    lo = params - 1.96 * bse
    hi = params + 1.96 * bse
    return pd.DataFrame({
        "model": label,
        "term": params.index,
        "coef": params.values,
        "robust_se": bse.values,
        "odds_ratio": np.exp(params.values),
        "or_ci_low": np.exp(lo.values),
        "or_ci_high": np.exp(hi.values),
        "p_value": p,
    })


def predict_with_ci(res, model_label: str, ratings=None) -> pd.DataFrame:
    """Fitted P(human) + 95% CI on the logit scale via the delta method."""
    if ratings is None:
        ratings = np.linspace(1, 10, 100)
    design_info = res.model.data.design_info
    cov = np.asarray(res.cov_params())
    out = []
    for author in ["AI", "Human"]:
        grid = pd.DataFrame({"rating": ratings, "true_author": author})
        X = np.asarray(dmatrix(design_info, grid))
        eta = X @ res.params.values
        var = np.einsum("ij,jk,ik->i", X, cov, X)
        se = np.sqrt(np.clip(var, 0, None))
        inv = lambda a: 1 / (1 + np.exp(-a))
        out.append(pd.DataFrame({
            "model": model_label, "true_author": author, "rating": ratings,
            "p_human": inv(eta), "ci_low": inv(eta - 1.96 * se),
            "ci_high": inv(eta + 1.96 * se),
        }))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", default=str(DEFAULT_SURVEY))
    ap.add_argument("--sheet_name", default=DEFAULT_SHEET)
    ap.add_argument("--wave_cutoff", default=DEFAULT_WAVE_CUTOFF)
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    tidy = build_tidy_df(args.input_xlsx, args.sheet_name, args.wave_cutoff)

    # Full sample
    res_full = fit_gee(tidy)
    coefs_full = coef_table(res_full, "full")
    coefs_full.to_csv(out_dir / "a2_craft_coefs_full.csv", index=False)

    # Restricted: AI at T <= 1.0 plus all human trials
    restricted = tidy[
        (tidy["true_author"] == "Human")
        | ((tidy["true_author"] == "AI") & (tidy["temperature"] <= 1.0))
    ].copy()
    res_restr = fit_gee(restricted)
    coefs_restr = coef_table(res_restr, "restricted")
    coefs_restr.to_csv(out_dir / "a2_craft_coefs_restricted.csv", index=False)

    preds = pd.concat([
        predict_with_ci(res_full, "full"),
        predict_with_ci(res_restr, "restricted"),
    ], ignore_index=True)
    preds.to_csv(out_dir / "a2_craft_predictions.csv", index=False)

    # --- Figure -------------------------------------------------------------
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"AI": "#d1495b", "Human": "#2e86ab"}
    for ax, mlabel, title in [
        (axes[0], "full", "Full sample"),
        (axes[1], "restricted", "Passing regime (AI T≤1.0 + humans)"),
    ]:
        for author in ["AI", "Human"]:
            d = preds[(preds["model"] == mlabel) & (preds["true_author"] == author)]
            ax.plot(d["rating"], d["p_human"], color=colors[author], label=author, lw=2)
            ax.fill_between(d["rating"], d["ci_low"], d["ci_high"],
                            color=colors[author], alpha=0.2)
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xlabel("Perceived quality rating")
        ax.set_title(title)
    axes[0].set_ylabel("P(judged “Human”)")
    axes[0].set_ylim(0, 1)
    axes[1].legend(title="True author", frameon=False)
    fig.suptitle("Craft heuristic: higher perceived quality → “human” verdict")
    fig.tight_layout()
    fig.savefig(out_dir / "a2_craft_heuristic.png", bbox_inches="tight")

    print("[A2] full-sample coefficients:")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(coefs_full.to_string(index=False))
        print("\n[A2] restricted (passing-regime) coefficients:")
        print(coefs_restr.to_string(index=False))
    print(f"\n[A2] wrote coefs + predictions + a2_craft_heuristic.png to {out_dir}")


if __name__ == "__main__":
    main()
