# Revision analyses

Self-contained scripts for the PNAS Nexus revision. Each writes one CSV of the
underlying numbers plus one figure into `../results/`. Repo-relative paths only
(no hard-coded `/Users/...`), fixed random seeds, wave-aware.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy statsmodels scikit-learn openpyxl matplotlib
```

## Run
```bash
cd analysis
../.venv/bin/python a1_detection_cis.py     # detection CIs + exact tests (~5 s)
../.venv/bin/python a2_craft_heuristic.py   # trial-level verdict model (~10 s)
../.venv/bin/python a3_homogenization.py    # full-sample homogenization (~8 min first run, then cached)
```

| Script | Roadmap item | Outputs |
|---|---|---|
| `a1_detection_cis.py` | A1 | `a1_detection_cis.csv`, `a1_detection_curve.png` |
| `a2_craft_heuristic.py` | A2 | `a2_craft_coefs_{full,restricted}.csv`, `a2_craft_predictions.csv`, `a2_craft_heuristic.png` |
| `a3_homogenization.py` | A3 | `a3_homogenization.csv`, `a3_homogenization.png` (+ `a3_cache/`) |
| `flitz_download.py` | new-flitz corpus | `flitz_manifest.csv` (+ downloads PDFs) |
| `flitz_extract.py` | new-flitz corpus | `flitz_survey2026_texts.xlsx`, `flitz_extracted.csv`, `flitz_file_map.csv` |
| `flitz_similarity.py` | new-flitz corpus | `flitz_similarity{,_matrix,_per_flitz,_heatmap}.*` |
| `plotHistogramDartvNonDart.py` | wave-2 institution split | `dart_vs_nondart_{scores,summary}.csv`, `dart_vs_nondart_histogram.png` |
| `common.py` | shared | wave-aware `build_tidy_df()`, stimulus map, paths, plot style |

### Detection data (Turing Survey.csv)
`Turing Survey.csv` is the re-opened-form export and holds **both** collection
periods in one file (117 rows from 2025 = wave 1, 20 rows from 2026 = wave 2),
separable by Timestamp; the 18 stimulus headers are byte-identical across years.
(`common.py` also accepts the earlier filename `Turing Survey2026.csv`.)
It's a graded-quiz export, so each question has extra `[Score]`/`[Feedback]`
columns — `common.py` strips those automatically. It also carries wave-2-only
covariates (flitzing familiarity, ethnicity, recruited-by).

Wave selection in A1:
```bash
../.venv/bin/python a1_detection_cis.py --wave 2 --suffix _wave2   # 2026 only (n=20)
../.venv/bin/python a1_detection_cis.py --wave 1 --suffix _wave1   # 2025 only
../.venv/bin/python a1_detection_cis.py                            # pooled + per-wave overlay
```

Wave-2 institution split (the "most recent educational institution" question
exists only in the 2026 re-open; 11 of the 20 respondents named Dartmouth, 9
named somewhere else):
```bash
../.venv/bin/python plotHistogramDartvNonDart.py   # side-by-side score histograms
```
Per-respondent score = correct verdicts out of 18, recomputed from the trial-level
judgments (matches the form's graded `Total score` exactly for all 20).
Dartmouth 13.4 ± 1.3 vs non-Dartmouth 12.7 ± 1.8; Mann-Whitney p = 0.51 — no
detectable difference, and n = 11/9 so this is descriptive only.

### New-flitz corpus (Flizting Survey2026.csv → human-human similarity)
The upload column holds Google Drive **links**, not files, and Forms uploads are
private, so automated download fails with a permission error. Get the files local
first (either set the Drive folder to "Anyone with the link" and re-run step 1,
or download the folder manually into `Data for Figures/flitz_pdfs/`), then:
```bash
../.venv/bin/python flitz_download.py     # 1. manifest + (attempt) download
../.venv/bin/python flitz_extract.py      # 2. uploads -> flitz text -> texts xlsx
../.venv/bin/python flitz_similarity.py   # 3. pairwise cosine + heatmap vs 33-flitz baseline
```
Extraction keeps only uploads with >= `--min_chars` of text (drops the "No Flitz"
blank pages). Gifs/memes are embedded images and are ignored — only text enters
the cosine, exactly as in the original pipeline.

The heatmap's diagonal is self-similarity and is 1 by construction; the reported
mean is over the off-diagonal pairs only. Step 2 emits rows in respondent order
(uploads are named after the sender, so raw file order is arbitrary), and step 3
labels the axes 1..7 in that order; the respondent each flitz came from is
recorded in `flitz_similarity_per_flitz.csv`. The 3 blank "No Flitz" uploads are
dropped in step 2 and never enter the cosine.

**Result (10 responses, Aug 2026):** 7 usable flitzes, 3 blank uploads.
Mean pairwise cosine **0.276** [0.210, 0.343] vs the 33-flitz baseline 0.264
[0.233, 0.298] — indistinguishable, but n=7 so the CI is wide; descriptive only.

What step 2 does beyond a raw text dump, and why:
- **Email chrome is stripped.** Most uploads are Gmail *print-to-PDF* files whose
  text layer carries the sender/recipient headers and a per-page print footer
  (`8/7/26, 7:48 PM` / `Dartmouth College Mail - <subject>` / `mail.google.com/...`
  / `1/3`). That boilerplate is *shared across documents*, so leaving it in
  inflates the pairwise cosine. Only the **first message's body** is kept — in a
  thread print, the recipient's reply and any follow-ups are dropped.
- **`.eml` uploads** are parsed as email (text/plain part, `[image: …]` GIF alt-text
  and the `-- ` signature block removed).
- **Uploads with no text layer** (a PDF that is only screenshots) are reported as
  `needs_ocr=True`, not silently dropped. `tesseract` is not installed here, so
  such a file needs a manual transcription in
  `Data for Figures/flitz_pdfs/transcriptions/<upload stem>.txt`, which then
  overrides extraction (`source = manual_transcription`; see that folder's README).
  `--ignore_transcriptions` reproduces the run from machine-extractable text only
  (n=6, mean 0.220 [0.097, 0.331]).
- **File → respondent mapping.** Uploads are named after the sender, not the
  response row, so files are matched to `flitz_manifest.csv` by mtime (= Drive
  creation time = submission time) against the Forms Timestamp. The two clocks
  differ by a constant offset, so both series are paired in time order, the offset
  is estimated as the median gap, and every residual must fall within
  `--match_tol_s` (here: −15.00 h offset, max residual 2 s, a clean 1:1 match).
  The map is cached to `results/flitz_file_map.csv`; `--remap` recomputes it.

## Wave 2 (2026 responses)

The 2026 responses come from the **same re-opened Google Form** and are separable
by date. When the export lands in `Data for Figures/`:

1. Verify stimulus identity: the 36 question headers (they contain the full flitz
   texts) must match wave 1 byte-for-byte.
2. Point the scripts at the combined export. `build_tidy_df()` assigns
   `wave` by `--wave_cutoff` (default `2026-01-01`): earlier = wave 1, later = wave 2.
3. A1 then emits `wave1` / `wave2` / `pooled` rows and overlays the per-wave
   curves automatically. A4/B1/B2 (wave separation, wave×temperature, heuristic
   drift) build on the same `wave` column.

No code change is needed for the wave split beyond passing the combined file.

## Method notes
- **Cosine is computed with a fresh `TfidfVectorizer` per pair** (IDF over the
  2-document pair). This is unusual but it is the *published* method — kept
  identical so A3 is comparable with Figs. 1/S2/S3.
- `human_written_flitzes.xlsx` has **no header row** — always read `header=None`.
- A2 models the **verdict** (`P(judged "Human")`), not `correct`, with
  respondent-cluster-robust SEs (GEE, exchangeable).
- A3 primary comparison is **cross-profile** within-AI pairs (like-for-like with
  the 33-distinct-author human baseline); same-profile and pooled are in the CSV
  for the SI. Caption caveat: near-zero cosine at high T = incoherence, not
  creative diversity.
