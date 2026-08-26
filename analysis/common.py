# -*- coding: utf-8 -*-
"""
Shared helpers for the revision analyses (A1, A2, A3, ...).

Everything here uses repo-relative default paths (no hard-coded /Users/... paths)
so the scripts run on any machine that has the repo checked out.

The key asset is `build_tidy_df()`: it turns the raw Google Forms export into a
trial-level table with one row per (respondent, flitz) judgment, adding a `wave`
column so that the SAME scripts work once the 2026 (wave-2) responses are added.

Wave assignment
---------------
The 2026 responses come from the SAME re-opened Google Form and are separable by
date. We assign wave by a timestamp cutoff (`--wave-cutoff`, default 2026-01-01):
    Timestamp <  cutoff  -> wave 1  (April 2025 collection)
    Timestamp >= cutoff  -> wave 2  (2026 collection)
With only the wave-1 file present today, every row is wave 1 and all pooled
analyses are identical to single-wave analyses. Nothing else changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- Repo layout -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data for Figures"
# The 2026 re-opened-form export: contains BOTH waves in one file (2025 rows =
# wave 1, 2026 rows = wave 2), separable by Timestamp. This is now the default.
# The export has been re-downloaded under a couple of names; take the first that
# exists so the scripts keep running after a re-export.
_SURVEY_CANDIDATES = ["Turing Survey.csv", "Turing Survey2026.csv"]
DEFAULT_SURVEY = next(
    (DATA_DIR / n for n in _SURVEY_CANDIDATES if (DATA_DIR / n).exists()),
    DATA_DIR / _SURVEY_CANDIDATES[0],
)
# The original wave-1-only export (quiz columns absent). Kept for reference.
WAVE1_ONLY_SURVEY = DATA_DIR / "04_12_ Distinguishing between AI vs. Human-written Flitzes (Responses).xlsx"
DEFAULT_SHEET = "Form Responses 1"
DEFAULT_RESULTS = REPO_ROOT / "results"
HUMAN_FLITZ_FILE = DATA_DIR / "human_written_flitzes.xlsx"
AI_SAMPLES_DIR = DATA_DIR / "ai_flitz_samples"

# Anything at/after this timestamp is treated as wave 2 (the 2026 re-open).
DEFAULT_WAVE_CUTOFF = "2026-01-01"

# --- Stimulus map (hard-coded in run_statistical_analysis.py) -----------------
HUMAN_FLITZ_NUMBERS = {1, 2, 6, 7, 8, 11, 12, 13, 16}
# AI flitz number -> generation temperature
TEMPERATURE_MAP = {
    3: 1.5, 4: 0.25, 5: 0.5, 9: 0.75, 10: 1.0,
    14: 1.25, 15: 0.0, 17: 1.75, 18: 2.0,
}
# Paired ordering used for the human/AI overlay figure (human flitz <-> temp).
HUMAN_FLITZ_ORDER = [1, 2, 12, 8, 6, 7, 11, 13, 16]
AI_FLITZ_ORDER    = [15, 4, 5, 9, 10, 14, 3, 17, 18]
SURVEYED_TEMPS    = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
# human flitz paired to each surveyed temperature (for the paired-FA option in B3)
HUMAN_PAIR_FOR_TEMP = dict(zip(SURVEYED_TEMPS, HUMAN_FLITZ_ORDER))


def _is_answer_col(c: str) -> bool:
    """True for a bare Forms answer column, False for quiz [Score]/[Feedback] ones.

    The 2026 export was collected as a graded quiz, so every question yields
    three columns: '<q>', '<q> [Score]', '<q> [Feedback]'. We keep only the bare
    answer column.
    """
    s = c.rstrip()
    return not (s.endswith("[Score]") or s.endswith("[Feedback]"))


def detect_columns(df: pd.DataFrame):
    prediction_columns = [c for c in df.columns
                          if c.startswith("Human or AI?") and _is_answer_col(c)]
    rating_columns = [c for c in df.columns
                      if c.startswith("Rate Flitz") and _is_answer_col(c)]
    if not prediction_columns:
        raise ValueError("No columns starting with 'Human or AI?' found.")
    if not rating_columns:
        raise ValueError("No columns starting with 'Rate Flitz' found.")
    return prediction_columns, rating_columns


def load_survey_frame(survey_path: Path | str, sheet_name: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Load either a .csv (2026 combined export) or an .xlsx (original) into a
    single DataFrame."""
    survey_path = Path(survey_path)
    if not survey_path.exists():
        raise FileNotFoundError(f"Survey file not found: {survey_path}")
    if survey_path.suffix.lower() == ".csv":
        return pd.read_csv(survey_path)
    book = pd.read_excel(survey_path, sheet_name=None)
    if sheet_name not in book:
        sheet_name = list(book.keys())[0]
    return book[sheet_name]


def build_tidy_df(
    survey_path: Path | str = DEFAULT_SURVEY,
    sheet_name: str = DEFAULT_SHEET,
    wave_cutoff: str = DEFAULT_WAVE_CUTOFF,
    min_answered: int = 2,
) -> pd.DataFrame:
    """Trial-level table with a wave column.

    Columns: respondent_id, timestamp, wave, flitz_number, true_author,
             predicted_author, correct, rating, temperature

    `min_answered` implements the paper's exclusion rule explicitly: respondents
    who answered fewer than `min_answered` verdict items are dropped (this is what
    turns the raw 117 into the analyzed 116). Set to 0 to disable.
    """
    df = load_survey_frame(survey_path, sheet_name).reset_index(drop=True)

    pred_cols, rating_cols = detect_columns(df)

    # tz-aware parse (2026 export carries mixed offsets), then drop tz for compare
    timestamps = pd.to_datetime(df.get("Timestamp"), errors="coerce", utc=True)
    if timestamps is not None:
        timestamps = timestamps.dt.tz_localize(None)
    cutoff = pd.Timestamp(wave_cutoff)

    # Explicit exclusion: count answered verdict items per respondent.
    answered_counts = df[pred_cols].notna().sum(axis=1)
    keep_mask = answered_counts >= min_answered

    rows = []
    for pred_col in pred_cols:
        m = re.search(r"Flitz (\d+)", pred_col)
        if not m:
            continue
        flitz_num = int(m.group(1))
        rating_col = next((c for c in rating_cols if f"Flitz {flitz_num}" in c), None)
        true_author = "Human" if flitz_num in HUMAN_FLITZ_NUMBERS else "AI"
        for idx, row in df.iterrows():
            if not keep_mask.iloc[idx]:
                continue
            pred = row[pred_col]
            if pd.isna(pred):
                continue
            rating = row[rating_col] if (rating_col and not pd.isna(row[rating_col])) else None
            predicted_author = str(pred).strip()
            ts = timestamps.iloc[idx] if timestamps is not None else pd.NaT
            wave = 2 if (pd.notna(ts) and ts >= cutoff) else 1
            rows.append({
                "respondent_id": idx,
                "timestamp": ts,
                "wave": wave,
                "flitz_number": flitz_num,
                "true_author": true_author,
                "predicted_author": predicted_author,
                "correct": int(predicted_author == true_author),
                "verdict_human": int(predicted_author == "Human"),
                "rating": rating,
            })

    tidy = pd.DataFrame(rows)
    tidy["temperature"] = tidy["flitz_number"].map(TEMPERATURE_MAP)
    return tidy


def find_institution_col(df: pd.DataFrame) -> str | None:
    """The wave-2-only 'most recent educational institution' free-text column."""
    for c in df.columns:
        if c.startswith("What is your most recent educational institution") and _is_answer_col(c):
            return c
    return None


def is_dartmouth(value) -> bool | None:
    """Classify the free-text institution answer as Dartmouth / elsewhere.

    Answers are free text ('Dartmouth College', 'Dartmouth', 'dartmouth college',
    'Dartmouth college Gbg', ...), so we match on the substring, case-insensitively.
    Returns None when the respondent left the question blank (all of wave 1: the
    question only exists in the 2026 re-open).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    return "dartmouth" in s.casefold()


# --- Plot style (kept close to make_figures.py) ------------------------------
def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def wave_label(w) -> str:
    return "pooled" if w == "pooled" else f"wave{w}"
