# Analysis With the Repo As It Stands
### Implementation assessment of Suggested_Analysis.md against `phasetransitions_networks_spinglasses_interactions_humanAI` (local clone of lhuynhm/SpinGlasses_LLMs_TuringPhaseTransitions)
*Inspected 15 Aug 2026. Verdict up front: everything in Tier A plus B3 is executable today with modest new scripts on top of existing repo assets; A4, B1, B2 are blocked on exactly one missing artifact — the wave-2 survey export, which is not in the repo. Tier C is dead: the survey never collected the fields.*

---

## 1. Repo inventory and the P1–P5 precondition verdicts

**What the repo contains.** Root: five cosine/similarity notebooks (`Cosine Similarity Code [save results/produce figures].ipynb`, `Pairwise Cosine Similarity.ipynb`, `AI-AI Pairwise Similarity Chart.ipynb`), four spin glass notebooks (`spinglassLLM_{phasetransition,clustering}_{saveresults,plot}.ipynb`), the survey pipeline `run_statistical_analysis.py` (byte-identical duplicate at `Data for Figures/run_analysis.py`), figure scripts (`make_figures_consistency_phasetransitions_data.py`, `Data for Figures/make_figures.py`), README. `Data for Figures/` holds all data: the survey export, the corpus, and precomputed similarity/spin-glass results.

**P1 — corpus and code: PASS.** The full frozen corpus is present. `human_written_flitzes.xlsx` = the 33 human flitzes; `ai_flitz_samples/` = 170 files (`temp_{T}_sample_{01–10}.xlsx`, 17 temperatures × 10 samples), each with 33 rows (`Prompt Index`, `Flitz Output`) → the full 5,610 generations, temperature-labeled by filename. Similarity code, survey code, and figure code are all present. Precomputed intermediates too (`average_cosine_similarity_results.xlsx` — 170 rows of per-file mean cosine; both Wasserstein tables; `energy_matrix.csv`, `gibbs_results.npy`, overlap CSVs).

**P2 — survey schema: AUDITED, and it kills Tier C.** `04_12_ Distinguishing between AI vs. Human-written Flitzes (Responses).xlsx`, sheet `Form Responses 1`: 117 rows × 38 columns = `Timestamp`, `Score` (Forms auto-grade), then 18 stimuli × (`Human or AI? …full flitz text in the header…`, `Rate Flitz k`). **No demographics, no graduation year, no flitzing familiarity, no AI-usage frequency, no free-text rationale.** Consequences: C1–C3 are unrunnable for wave 1 (only possible if the wave-2 form added fields — check when the export arrives); B1's composition adjustment cannot use covariates, so the composition-confound caveat moves from "adjust where possible" to mandatory unqualified language in the text.

**P3 — respondent reconciliation: RESOLVED.** The file has exactly 117 responses, timestamps 2025-04-10 → 2025-04-24 (so "04_12" is the form date, not the export coverage). The paper's 116 comes from excluding one respondent who answered only 1 question. Note: `run_statistical_analysis.py` never applies that exclusion explicitly — it just drops NaN cells while building the trial-level table, which handles it implicitly at the trial level but means "n = 116" is nowhere enforced in code. Fix in A5: add an explicit exclusion rule (e.g., respondents with < k answered items) and have every output CSV carry its own N.

**P4 — stimulus map: RESOLVED, it was in the code all along.** `run_statistical_analysis.py` hard-codes it: human flitzes = {1, 2, 6, 7, 8, 11, 12, 13, 16}; AI flitzes with temperatures = {15: 0.00, 4: 0.25, 5: 0.50, 9: 0.75, 10: 1.00, 14: 1.25, 3: 1.50, 17: 1.75, 18: 2.00}; and a human↔AI pairing order used for the overlay figure. All 18 stimuli were shown to every respondent, so each temperature point rests on ~116 judgments of **one single AI flitz** — worth stating in Methods, since it means the detection curve confounds temperature with the identity of the sampled flitz (a per-stimulus, not per-temperature, design). The CIs in A1 are computed over respondents accordingly.

**P5 — wave metadata: BLOCKED.** There is **no wave-2 survey export anywhere in the repo** — the single biggest gap between the plan and the repo. The one survey file predates the redone survey by ~16 months. Until someone exports the 2026 responses from Google Forms, A4, B1, and B2 cannot start. Two sub-cases once it arrives: (a) same form re-opened → one export contains both waves, separable cleanly by `Timestamp` (2025-04 cluster vs 2026 cluster); (b) a cloned form → separate file, and stimulus identity should be verified by comparing the 36 question headers byte-for-byte between exports (the headers contain the full flitz texts, so this check is free and *is* the P5 stimulus-identity audit).

---

## 2. What can be done with the repo exactly as it is (and how)

The repo's underappreciated asset is `run_statistical_analysis.py`'s `build_tidy_df()`: it already converts the raw Forms export into a trial-level table (`respondent_id`, `flitz_number`, `true_author`, `predicted_author`, `correct`, `rating`, `temperature`) and writes `results/tidy_df.csv`. That table is precisely the input every survey-side analysis in the plan needs. Nothing has to be re-derived from raw data.

**A1 — detection CIs and exact tests: runnable now (wave-1 only).**
How: load `tidy_df.csv`, group AI trials by temperature, n and successes per group → Wilson 95% CI (`statsmodels.stats.proportion.proportion_confint(method='wilson')`), exact binomial test vs 0.5 (`scipy.stats.binomtest`), Holm across the 9 temperatures. ~60 lines in a new `analysis/a1_detection_cis.py`; writes the CSV and the error-bar figure (style block copy-pasted from `make_figures.py`). With n ≈ 116 per temperature, chance-level accuracy has a CI of roughly ±9 points — so expect the T = 1.25 dip (41%) to be statistically indistinguishable from 0.5, as anticipated; text softening applies.

**A2 — trial-level craft-heuristic model: runnable now, with two deliberate changes to what the existing code does.**
The repo's `compute_logistic_predictions()` already fits trial-level logistics (so the situation is slightly better than the roadmap feared), but (i) it models `correct`, whereas the craft heuristic is about the *verdict* — outcome must be `predicted_author == 'Human'`; and (ii) it ignores respondent clustering entirely, and the figure's curves carry no uncertainty. How: refit as logistic with respondent-clustered robust SEs or GEE (`statsmodels` GEE, exchangeable working correlation, cluster = `respondent_id`) — both already-installed dependencies; a full random-effects GLMM adds little here and would be the only thing requiring new tooling, so cluster-robust GEE is the pragmatic, defensible choice. Then the restricted refit: subset to AI trials with T ≤ 1.0 plus all human trials. New `analysis/a2_craft_heuristic.py`; regenerates the Fig. 3B successor with CI bands from the model.

**A3 — full-sample homogenization: runnable now; the corpus and the pairwise function both exist.**
How: reuse the per-pair TF-IDF cosine helper that `Pairwise Cosine Similarity.ipynb` already defines (`cosine_matrix_pairwise()`), applied per temperature to all 330 AI flitzes (10 sample files × 33 rows, loaded the way `Cosine Similarity Code [save results].ipynb` loads them), against the within-human baseline over the 33 human flitzes (528 pairs). Scale check: C(330,2) = 54,285 pairs × 17 temperatures ≈ 923k tiny vectorizer fits — roughly 30–60 min single-core, embarrassingly parallelizable over temperatures if needed; entirely feasible.
One substantive design decision the plan should make explicit: within-AI pairs come in two strata — same-profile pairs (two generations from the *same* virtual student, expected similar by construction) and cross-profile pairs. Report **cross-profile pairs as the primary homogenization comparison** (that is the like-for-like analogue of the 33-author human baseline: 33 profiles, one flitz each drawn per resample) with the pooled version in SI; otherwise a reviewer can attribute the homogeneity to prompt identity rather than to the model flattening the genre. Matched-size bootstrap (sample 33 AI flitzes, one per profile, per iteration) handles both this and the CI in one loop. New `analysis/a3_homogenization.py`.

**B3 — d′ and criterion per temperature: runnable now.**
How: from `tidy_df.csv`, hit rate = P(judged AI | AI flitz at T); false-alarm rate = P(judged AI | human flitz). Human stimuli carry no temperature, so two options: pooled FA rate across all 9 human flitzes (one criterion baseline, cleanest), or the per-temperature paired human flitz via the pairing order hard-coded in `run_statistical_analysis.py` (`human_flitz_order` ↔ `ai_flitz_order`) — compute both, headline the pooled version, SI the paired one. Log-linear correction for extreme rates (91% accuracy cells). New `analysis/b3_signal_detection.py`; candidate replacement panel for the |accuracy − 0.5| figure.

**A5 — reporting hygiene: mostly runnable now.** Per-temperature judgment counts, explicit exclusion rule, per-wave N (wave 1 for now) all fall out of the tidy table; the `Score` column doubles as a free cross-check of the correctness coding (Score should equal each respondent's sum of `correct`). The biased/unbiased renaming and Methods sentences live in the LaTeX, not this repo.

**Spin glass side: nothing to do, confirmed.** The four notebooks are self-contained, all their precomputed outputs are already in `Data for Figures/`, and the paper draft needs no new simulation. (For the record: the phase-transition simulation hard-codes `n_people = 116`, consistent with the analyzed survey N.)

---

## 3. What must be added or changed

**Blocking addition — the wave-2 export (not a code problem).** Nothing in Tier B runs until the 2026 Google Forms responses land in `Data for Figures/` as an xlsx with the same schema. Whoever owns the Form needs to export it; this is the item to escalate to the senior author *today* if access is unclear. On arrival: run the header byte-comparison (P5), then A4 → B1 → B2.

**Code changes, in order of importance:**

1. **Wave-aware tidy builder** (`run_statistical_analysis.py`): accept multiple input files (or one file spanning both waves), assign `wave` by source file or by `Timestamp` cutoff, emit it as a column in `tidy_df.csv`. ~20 lines. Every downstream script then gets per-wave/pooled behavior for free by grouping. This unblocks A4 (per-wave curves + equivalence tests), B1 (`correct ~ wave × temperature` logistic with clustered SEs — same machinery as A2), and B2 (A2's model + wave interaction; quality-drift comparison) the moment data arrives.
2. **Kill the hard-coded paths.** Every cosine notebook reads from `/Users/jacksongeorge/Desktop/...`; none of them run on your machine as-is. New analysis scripts should take repo-relative paths (`Data for Figures/...`) with `argparse` defaults, like `run_statistical_analysis.py` already does.
3. **Mind the two data traps.** (a) `human_written_flitzes.xlsx` has no header row — read with `header=None` as the [save results] notebook correctly does, or pandas silently swallows flitz #1 as the header and you analyze 32 texts (loading it naïvely yields shape (32, 1)). (b) `Pairwise Cosine Similarity.ipynb` expects a file named `real_flitzes.xlsx` that does not exist in the repo — it is the same file as `human_written_flitzes.xlsx`; point it there.
4. **Reproducibility fixes while touching the code:** the AI-AI subsampling notebook runs with `RANDOM_SEED = None` (its published heatmaps are literally non-reproducible — one more reason A3 supersedes it); set seeds in every new script and keep the `versions.json` habit from `run_analysis.py`. Keep the one-script-per-task + figure + numbers-CSV layout from the roadmap so the senior author can check numbers without running anything.
5. **Consistency constraint, not a change:** the pipeline's cosine is computed with a *fresh TfidfVectorizer per pair* (IDF over a 2-document corpus). That is unusual but it is the published method — A3 and any new similarity work must reuse the same per-pair scheme, not a corpus-wide vectorizer, or the new numbers won't be comparable with Figs. 1/S2/S3 and the method text.

**What cannot be fixed with code:** Tier C (fields never collected — drop it, unless the wave-2 form differs), and the per-temperature single-stimulus design (a limitation sentence, not an analysis).

---

## 4. Bottom line

| Plan item | Status with current repo | What it takes |
|---|---|---|
| A1 detection CIs | ✅ runnable today | new ~60-line script on `tidy_df.csv` |
| A2 craft heuristic (trial-level + restricted) | ✅ runnable today | new script; switch outcome to verdict, add clustered SEs |
| A3 homogenization (full sample) | ✅ runnable today | new script reusing existing pairwise helper; ~30–60 min compute; cross-profile stratum as primary |
| B3 d′/criterion | ✅ runnable today | new small script on `tidy_df.csv` |
| A5 hygiene | ✅ mostly today | counts + exclusion rule; text edits live in the LaTeX |
| A4 wave separation | ⛔ blocked | wave-2 export + 20-line wave-aware tidy builder |
| B1 wave × temperature | ⛔ blocked | same; then same machinery as A2 |
| B2 heuristic drift | ⛔ blocked | same |
| C1–C3 | ❌ dead for wave 1 | fields don't exist; re-check wave-2 schema on arrival |
| Spin glass | ✅ nothing needed | precomputed outputs already in repo |

The rational sequence given the blocker: start A1/A2/A3/B3 immediately on wave-1 data (they are wave-independent in their pooled form and their scripts will re-run unchanged once the wave column exists), request the wave-2 export in parallel, and slot in A4 → B1 → B2 the day it lands.
