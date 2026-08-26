# Suggested Analysis for the PNAS Nexus Revision
### Based on: May 2026 submission, PNAS Nexus draft (11 Aug 2026, v4), and the data-analysis roadmap
*Constraint set: methods stay as they are (TF-IDF/cosine + Wasserstein, survey, coupled spin glass; frozen corpus of 33 human flitzes + 5,610 GPT-4o generations). All analysis must be executable in a few days. Deadline: 1 Sept 2026.*

---

## 1. Where the revised paper stands, analysis-wise

The PNAS Nexus draft has committed to a stronger story than the May submission actually supports statistically. Three headline claims now carry the paper:

1. **Structural transition + homogenization** — "within the conventional regime, the machine's flitzes are markedly more homogeneous than the human corpus" (Abstract, Significance, Section 1.1, Discussion).
2. **Reception tracks the boundary** — detection at chance below the transition, 41% local minimum at T = 1.25, 91% at T = 1.75.
3. **Craft heuristic** — perceived quality predicts a "human" verdict; the passing register is the most flattened one.

All three currently rest on point estimates, aggregated fits, or a 5-flitz subsample. The draft's own editor notes concede as much ("add quantitative homogeneity numbers in Section 1.1"; "decide n = 116 vs. 117"; "regenerate the recomposed structure/reception figures"). Meanwhile the draft's blue note names exactly what the new survey wave should deliver: *"(2) this year's scores are higher than last year's scores, implying some kind social evolution pattern."* That claim is currently written down nowhere as an analysis — it is the single most valuable thing the redone survey buys, and the strongest possible fit to the Culture & AI call (it turns "AI influencing culture" from speculation into a measured reception change on frozen stimuli).

The roadmap attached to this revision is, in my assessment, still essentially correct even though it was written for an earlier framing — its tiers map cleanly onto the current draft. Below I re-evaluate it against the *current* draft: what to keep, what to re-prioritize, what it misses, and what can be dropped under the time constraint.

---

## 2. Required — the paper is not review-safe without these (Tier A, confirmed and sharpened)

### A1. Uncertainty on the detection curve (Fig. 2A / Fig. 3A successor)
Wilson 95% CIs per temperature; exact binomial tests against 0.5; Holm correction across the 9 surveyed temperatures. Run per wave and pooled (pooling only after A4).
- **Why it matters now:** every reception claim in Results 1.2 and the Significance statement is a point estimate. With ~116 respondents split over 9 stimuli, the 41% "local minimum" at T = 1.25 is within ~2 SE of chance and will probably not survive; the 91% at T = 1.75 trivially will. The text must be ready to soften "a local minimum occurs" to "accuracy is at or below chance," which — importantly — does not weaken the story: the phase-transition claim needs *chance below the boundary, high accuracy above it*, not a significant dip.
- **Lands in:** error bars on the detection panel; one SI table (N, accuracy, CI, p per temperature); softened sentence in 1.2 if the dip dies.

### A2. Trial-level craft-heuristic model (replaces the Fig. 3B fit)
Mixed-effects logistic regression on individual responses: verdict("human") ~ rating, random intercept per respondent (per stimulus if estimable), separate slopes for human-written vs AI-generated stimuli; odds ratios + CIs. Then the **restricted refit on the passing regime only** (AI at T ≤ 1.0 + all human flitzes).
- **Why it matters now:** Fig. 3B fits aggregated means per rating bin — an ecological-fallacy exposure with no clustering correction, and any social-science reviewer at PNAS Nexus will flag it on first read. The restricted version is the paper's key *unverified* claim: the intro and Discussion both assert the heuristic operates even where detection is at chance (i.e., it is decoupled from mere incoherence detection). That sentence is currently gated on an analysis nobody has run.
- **Lands in:** regenerated craft-heuristic figure with model-based curves + CI bands; restricted-regime panel or SI figure; ORs in text.

### A3. Homogenization quantified on the full sample (the draft's own open TODO)
All pairwise within-AI cosine similarities per temperature (330 flitzes per temperature → full pair set), against the within-human baseline (33 flitzes, 528 pairs). Bootstrap CIs; matched-size resampling; report split below vs above the transition.
- **Why it matters now:** homogenization is in the title logic, the Significance statement, and the Abstract ("markedly more homogeneous"), but the evidence is Appendix D's 5-randomly-selected-flitz heatmaps (Figs. S2/S3) and a qualitative sentence. This is the largest gap between claim strength and evidence strength in the paper. Also mandatory: the caption caveat that near-zero cosine at high T reflects vocabulary disjointness from incoherence, *not* healthy creative diversity — otherwise Fig. S3's dark-blue panels invite exactly the wrong reading.
- **Lands in:** one main-text figure (within-human vs within-AI similarity distributions by temperature, passing regime highlighted); the promoted quantitative numbers in Section 1.1; SI keeps the heatmaps.

### A4. Wave separation before any pooling (new obligation created by the redone survey)
Reproduce detection and quality curves per wave; test wave equivalence (two-proportion tests per temperature, or logistic with wave main effect) *before* pooling. Equivalent → pool with stated justification; not equivalent → report separately and B1 becomes a headline.
- **Why it matters now:** the moment wave-2 responses enter any figure, silent pooling of two collection periods ~1.5 years apart is a self-inflicted reviewer objection. And the per-wave replication of the curve on frozen stimuli is a free robustness sentence regardless of outcome.
- **Lands in:** SI figure (overlaid per-wave curves); one sentence in Methods, one in Results.

### A5. Reporting hygiene (cheap, mandatory, mostly bookkeeping)
- Resolve **n = 116 vs 117** (the draft's own note) and state per-wave N, populations (on-campus vs outside), exclusion rule.
- Per-temperature judgment counts (which 9 pairs, which temperatures, how many judgments each) — needed by A1's CIs anyway.
- Quality-rating figure (Fig. 2B): replace ±1 SD shading with SE or 95% CI of the mean — SD shading visually overstates uncertainty about the *mean* while understating response spread information; at minimum state which it is and why.
- Rename "biased/unbiased data" framing (survives in Fig. 2's caption title "biased and unbiased data") → human-judgment vs text-structure data.
- Every statistic that lands in the text: N, estimate, CI, test, correction. No new point estimates anywhere.

---

## 3. High value — the new wave's actual payoff (Tier B, re-ordered for the current draft)

### B1. Wave × temperature interaction on detection — **the single most valuable addition**
Logistic regression: correct-detection ~ wave × temperature (+ respondent covariates where the survey schema has them). Frozen stimuli make this a clean reception-change design: identical texts judged ~1.5 years apart.
- Either direction is a finding: detection improved (the community has learned the machine's tells) or worsened (AI-inflected prose has normalized). This is precisely the draft's blue-note conjecture "(2)", and it upgrades the paper's fit to the call's *influencing* verb from prediction to observation.
- **Mandatory caveat wherever reported:** waves differ in composition, not just time. Adjust for covariates where possible; claim "reception difference consistent with cultural change," never "cultural change."
- **Lands in:** one figure (per-wave curves + interaction estimate) — main text if significant, SI otherwise; a Discussion paragraph either way.

### B3 (promoted above B2). Signal-detection reframing: d′ and criterion per temperature
Split accuracy into hit rate (AI judged AI) and false-alarm rate (human judged AI); compute d′ and criterion c per temperature, per wave.
- Best effort-to-elegance ratio in the whole plan. It cleanly restates the craft heuristic as a *criterion shift* and the transition as a *discrimination jump*, and it can replace the weakest panel in the paper — Fig. 3A's |accuracy − 0.5| coin-deviation framing with its "Law of Large Numbers" digression, which a statistics-literate reviewer will find odd (the LLN analogy to the partition function is decorative, not inferential; it is already hedged with "Curiously" in the draft).
- **Lands in:** candidate replacement for Fig. 3A (old version to SI); rewrite of the opening of Results 1.3.

### B2. Wave comparison of the craft heuristic and quality ratings
(i) A2's model with a wave interaction: has the quality→human slope changed? (ii) Perceived quality of passing-regime AI flitzes by wave: quality-drift.
- Directly tests "the heuristic that admits the machine is strengthening/weakening"; feeds the same Discussion paragraph as B1. SI figure/table + one or two sentences.

---

## 4. Opportunistic — only if the survey schema has the fields (Tier C, unchanged)

- **C1.** Graduation year / age band as detection moderator (exposure vs normalization framing) — exploratory, underpowered, SI at most.
- **C2.** Flitzing familiarity, AI-usage frequency, community membership as covariates in B1 (this doubles as the wave-composition adjustment — so check for these fields *first*; C2 is the only C item that feeds a Tier B analysis).
- **C3.** Free-text rationales, if collected: lightweight coding of stated cues. Optional texture for "how the community authenticates."

---

## 5. What does *not* need doing (explicitly out of scope)

- **No new GPT-4o generation.** The frozen corpus is a stated feature (identical stimuli across waves is what makes B1 clean).
- **No changes to the spin glass model or simulations.** The model section is internally consistent and its claims are appropriately qualitative ("consistent with"). One optional cheap robustness note: the 36-cell Shapiro–Wilk grid in Fig. S1 involves 36 implicit tests — a one-sentence multiple-comparison acknowledgment in the caption suffices; re-running simulations is not needed.
- **No new similarity metrics, no embedding models.** Same TF-IDF pipeline throughout — method consistency across waves and sections is worth more than metric novelty here.
- **No restoration of the old math-paper framing.** Appendix E stays parked; nothing below requires the departed co-authors' input except where a data artifact turns out to be missing from the repo.

---

## 6. Preconditions to verify on the repo before anything runs (P1–P5)

These are unchanged from the roadmap and remain blocking:

- **P1** Repo inventory: corpus (human + generated with temperature labels), survey export(s) for *both* waves, similarity code, figure scripts.
- **P2** Survey schema audit for both waves: every collected field (affiliation, graduation year, flitzing familiarity, AI-usage, free text, timestamps). Decides C-tier and the B1 covariate adjustment.
- **P3** Respondent reconciliation: 117 collected vs 116 analyzed vs "~120" in notes; per-wave N; exclusion rule; populations.
- **P4** Stimulus map: which 9 pairs at which temperatures, per-temperature judgment counts.
- **P5** Wave metadata: wave-2 stimuli byte-identical to wave 1; responses carry a wave identifier or are separable by date. **If waves are not separable, Tier B is dead** and all pooled analyses must be labeled as pooled.

---

## 7. Suggested execution order (a few days)

| Day | Work |
|---|---|
| 1 | P1–P5 on the repo; A5 bookkeeping resolved on paper (n, populations, stimulus map) |
| 1–2 | A1 (CIs + exact tests) and A4 (wave separation/equivalence) — A4 before anything pools |
| 2–3 | A2 (trial-level model + restricted-regime refit); A3 (full-sample homogenization + bootstrap) |
| 3–4 | B3 (d′/criterion), B1 (wave × temperature), B2 (heuristic drift) |
| if time | C2 → C1 → C3, in that order |

Every task writes one script + one figure + one small CSV of the underlying numbers (the CSVs are what get pasted into the paper and checked by the senior author). Figures regenerated at journal resolution while touched; the 2×2 combined figure splits A/B vs C/D per the restructure TODO.

## 8. Bottom line

Tier A converts the three existing headline results into review-safe form and closes the draft's own open TODOs — it defends the story already written, and A2's restricted refit gates a sentence the intro currently assumes. B1 is the new wave's payoff and the paper's strongest possible upgrade for this call; B3 is the cheapest elegance win and fixes the draft's weakest panel. If time collapses: all of Tier A is non-negotiable, then B1, then B3. Everything else can be dropped without endangering the submission.
