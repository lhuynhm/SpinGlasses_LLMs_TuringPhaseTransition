# run_analysis.py
# -*- coding: utf-8 -*-
"""
Re-run analysis, generate ALL derived datasets used by figures, and save to /results.
No plotting here. The make_figures.py script will load these files and reproduce figures exactly.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import statsmodels.api as sm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_columns(df: pd.DataFrame):
    prediction_columns = [col for col in df.columns if col.startswith("Human or AI?")]
    rating_columns = [col for col in df.columns if col.startswith("Rate Flitz")]
    if not prediction_columns:
        raise ValueError("No columns starting with 'Human or AI?' found.")
    if not rating_columns:
        raise ValueError("No columns starting with 'Rate Flitz' found.")
    return prediction_columns, rating_columns

def build_tidy_df(df: pd.DataFrame, prediction_columns, rating_columns):
    # Human flitz IDs
    human_flitz_numbers = {1, 2, 6, 7, 8, 11, 12, 13, 16}

    rows = []
    for pred_col in prediction_columns:
        m = re.search(r'Flitz (\d+):', pred_col)
        if not m:
            continue
        flitz_num = int(m.group(1))
        rating_col = next((c for c in rating_columns if f'Flitz {flitz_num}' in c), None)
        if rating_col is None:
            continue
        for idx, row in df.iterrows():
            pred = row[pred_col]
            if pd.isna(pred):
                continue
            rating = row[rating_col] if not pd.isna(row[rating_col]) else None
            true_author = 'Human' if flitz_num in human_flitz_numbers else 'AI'
            predicted_author = str(pred).strip()
            rows.append({
                'respondent_id': idx,
                'flitz_number': flitz_num,
                'true_author': true_author,
                'predicted_author': predicted_author,
                'correct': int(predicted_author == true_author),
                'rating': rating
            })
    tidy = pd.DataFrame(rows)
    # Attach temperature map for AI flitzes
    temperature_map = {
        3: 1.5, 4: 0.25, 5: 0.5, 9: 0.75, 10: 1.0,
        14: 1.25, 15: 0.0, 17: 1.75, 18: 2.0
    }
    tidy['temperature'] = tidy['flitz_number'].map(temperature_map)
    return tidy

def save_versions(results_dir: Path):
    import numpy
    import pandas
    try:
        import seaborn
        seaborn_ver = seaborn.__version__
    except Exception:
        seaborn_ver = None
    import matplotlib
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn_ver,
        "statsmodels": sm.__version__,
        "scipy": None
    }
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    (results_dir / "versions.json").write_text(json.dumps(versions, indent=2))

def local_extrema(y_vals: np.ndarray):
    # Return lists of indices for local maxima/minima
    max_idx = argrelextrema(y_vals, np.greater)[0].tolist()
    min_idx = argrelextrema(y_vals, np.less)[0].tolist()
    return max_idx, min_idx

def compute_accuracy_by_temp(tidy_df: pd.DataFrame):
    acc = tidy_df.groupby('temperature', dropna=True)['correct'].mean().reset_index().rename(columns={'correct':'accuracy'})
    flitz_labels = {0.0:'15', 0.25:'4', 0.5:'5', 0.75:'9', 1.0:'10', 1.25:'14', 1.5:'3', 1.75:'17', 2.0:'18'}
    acc['flitz_labels'] = acc['temperature'].map(flitz_labels)
    return acc

def compute_ratings_by_temp(tidy_df: pd.DataFrame):
    ai_rated_df = tidy_df[
        (tidy_df['true_author'] == 'AI') &
        (tidy_df['temperature'].notna()) &
        (tidy_df['rating'].notna())
    ].copy()
    ratings = (ai_rated_df.groupby('temperature')['rating']
               .agg(['mean','std','count'])
               .reset_index()
               .rename(columns={'mean':'avg_rating','std':'std_dev'}))
    return ratings

def compute_human_accuracy_by_flitz(tidy_df: pd.DataFrame):
    df = tidy_df[tidy_df['true_author'] == 'Human'] \
        .groupby('flitz_number')['correct'].mean().reset_index() \
        .rename(columns={'flitz_number':'flitz_id', 'correct':'accuracy'})
    df['flitz_label'] = df['flitz_id'].astype(str)
    return df

def compute_binned_means_for_fig3(tidy_df: pd.DataFrame, include_lowest: bool = False):
    # include_lowest=False keeps the original behaviour (ratings of exactly 1 fall
    # outside the first bin and are dropped).  The LLM-judge panels pass True,
    # because a rating of 1 is a very common verdict there.
    bins = np.arange(1, 11.5, 0.5)
    labels = (bins[:-1] + bins[1:]) / 2
    out_rows = []
    for author_type in ["AI", "Human"]:
        subset = tidy_df[tidy_df["true_author"] == author_type].copy()
        subset["correct_numeric"] = subset["correct"].astype(int)
        subset["rating_bin"] = pd.cut(subset["rating"], bins=bins, labels=labels,
                                      include_lowest=include_lowest)
        grouped = subset.groupby("rating_bin", observed=False)["correct_numeric"]
        mean_correct = grouped.mean()
        count = grouped.count()
        sem_correct = grouped.std() / np.sqrt(count)
        tmp = pd.DataFrame({
            "author_type": author_type,
            "rating_bin_mid": mean_correct.index.astype(float),
            "mean_correct": mean_correct.values,
            "sem_correct": sem_correct.values,
            "n": count.values
        })
        out_rows.append(tmp)
    return pd.concat(out_rows, ignore_index=True)

def compute_logistic_predictions(tidy_df: pd.DataFrame, n_points: int = 100):
    rating_range = np.linspace(1, 10, n_points)
    all_preds = []
    for author_type in ["AI", "Human"]:
        subset = tidy_df[tidy_df["true_author"] == author_type].copy()
        subset["correct_numeric"] = subset["correct"].astype(int)
        subset = subset.dropna(subset=["rating", "correct_numeric"])
        if subset.empty:
            continue
        y = subset["correct_numeric"]
        if y.nunique() < 2:
            # Degenerate outcome (a judge that is right -- or wrong -- everywhere):
            # the logistic model is not identified, its MLE is the constant fit.
            y_pred = np.full(n_points, float(y.iloc[0]))
        else:
            X = sm.add_constant(subset["rating"])
            model = sm.Logit(y, X).fit(disp=False)
            X_pred = sm.add_constant(rating_range)
            y_pred = model.predict(X_pred)
        df_pred = pd.DataFrame({
            "Author": author_type,
            "Rating": rating_range,
            "Predicted_Probability": y_pred
        })
        all_preds.append(df_pred)
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame(columns=["Author","Rating","Predicted_Probability"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", type=str, default="Form_Responses.xlsx", help="Path to Excel file")
    parser.add_argument("--sheet_name", type=str, default="Form Responses 1", help="Excel sheet name")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to write result files")
    args = parser.parse_args()

    in_path = Path(args.input_xlsx)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    results_dir = Path(args.out_dir)
    ensure_dir(results_dir)

    # Load data
    excel_data = pd.read_excel(in_path, sheet_name=None)
    if args.sheet_name not in excel_data:
        raise ValueError(f"Sheet '{args.sheet_name}' not found in workbook.")
    df = excel_data[args.sheet_name]

    # Parse / tidy
    pred_cols, rating_cols = detect_columns(df)
    tidy_df = build_tidy_df(df, pred_cols, rating_cols)

    # Save tidy_df
    tidy_df.to_csv(results_dir / "tidy_df.csv", index=False)

    # Accuracy by temperature
    acc_temp = compute_accuracy_by_temp(tidy_df).sort_values("temperature")
    acc_temp.to_csv(results_dir / "accuracy_by_temp_labeled.csv", index=False)

    # Extrema for fig1A
    y_acc = acc_temp['accuracy'].to_numpy()
    max_idx, min_idx = local_extrema(y_acc)
    meta_1A = {
        "local_max_indices": max_idx,
        "local_min_indices": min_idx
    }
    (results_dir / "figure1A_extrema.json").write_text(json.dumps(meta_1A, indent=2))

    # Ratings by temperature (mean/std)
    ratings_by_temp = compute_ratings_by_temp(tidy_df).sort_values("temperature")
    ratings_by_temp.to_csv(results_dir / "ratings_by_temp.csv", index=False)
    # Extrema for fig2 (on avg rating)
    y_rate = ratings_by_temp['avg_rating'].to_numpy()
    max_idx2, min_idx2 = local_extrema(y_rate)
    meta_2 = {
        "local_max_indices": max_idx2,
        "local_min_indices": min_idx2
    }
    (results_dir / "figure2_extrema.json").write_text(json.dumps(meta_2, indent=2))

    # Fig3 precomputations
    bins_stats = compute_binned_means_for_fig3(tidy_df)
    bins_stats.to_csv(results_dir / "fig3_binned_means_se.csv", index=False)

    logistic_preds = compute_logistic_predictions(tidy_df, n_points=100)
    logistic_preds.to_csv(results_dir / "logistic_fit_predictions.csv", index=False)

    # Fig4: Human flitz accuracy
    human_acc = compute_human_accuracy_by_flitz(tidy_df)
    human_acc.to_csv(results_dir / "accuracy_by_flitz_human.csv", index=False)

    y_hacc = human_acc['accuracy'].to_numpy()
    max_idx4, min_idx4 = local_extrema(y_hacc)
    meta_4 = {
        "local_max_indices": max_idx4,
        "local_min_indices": min_idx4
    }
    (results_dir / "figure4_extrema.json").write_text(json.dumps(meta_4, indent=2))

    # Fig5 overlay data
    human_flitz_order = [1, 2, 12, 8, 6, 7, 11, 13, 16]
    ai_flitz_order    = [15, 4,  5,  9, 10,14,  3, 17, 18]
    temps             = [0.0, 0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]

    # Build human accuracy lookup
    human_subset = tidy_df[(tidy_df['true_author']=='Human') & (tidy_df['flitz_number'].isin(human_flitz_order))]
    human_lookup = human_subset.groupby('flitz_number')['correct'].mean().to_dict()
    human_acc_seq = [human_lookup.get(fid, np.nan) for fid in human_flitz_order]

    # AI accuracy from acc_temp (ensure sorted by temp)
    ai_acc_seq = acc_temp.set_index("temperature").loc[temps, "accuracy"].to_list()

    overlay = pd.DataFrame({
        "temperature": temps,
        "ai_acc": ai_acc_seq,
        "human_acc": human_acc_seq,
        "human_flitz_order": human_flitz_order,
        "ai_flitz_order": ai_flitz_order
    })
    overlay.to_csv(results_dir / "figure5_overlay_data.csv", index=False)

    # Save mapping/meta and library versions
    meta = {
        "input_file": str(in_path),
        "sheet_name": args.sheet_name,
        "human_flitz_order": human_flitz_order,
        "ai_flitz_order": ai_flitz_order,
        "temps": temps
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    save_versions(results_dir)

    print(f"Done. Wrote derived datasets to: {results_dir.resolve()}")

if __name__ == "__main__":
    main()
