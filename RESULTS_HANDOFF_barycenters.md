# Barycenter Baseline Comparison — Results Handoff

**Run on:** the Smartflat data server (`SMARTFLAT_DATA_ROOT=/home/perochon/data-gold-final`), round 8, `SymbolicSourceInferenceGoldConfig`.
**Scope:** fix and run NB06/NB06b end-to-end; re-select TW-TWE hyperparameters; produce the baseline comparison + ablation + significance + stability for the paper.
**Headline:** the pipeline now runs end-to-end, but a faithful reproduction surfaces a substantive finding that bears on the paper's §5.5 claim — documented below. **No results were tuned to make the proposed method win.**

> ⚠️ **Flag for the authors (do not silently integrate):** on this data, with the K-space symbolic representation and a faithful pipeline, the proposed **TW-TWE + DBA** method is **competitive only on RIL-vs-Control** and is **matched or beaten by simple frequency baselines (Wasserstein histogram, majority voting)** on the pooled Patient-vs-Control comparison. The group-discriminative signal is dominated by **which prototypes occur (symbol frequency)**, not their temporal alignment. See §3.

> 🔄 **UPDATE (2026-06-05) — G=28 reproduction + audit + beat-MV experiments (notebooks `06c`/`06d`, new module `vocab.py`):** the coarse **G=28 semantic-category** vocabulary (27 named cooking actions + background) was built end-to-end from the raw labels via the pyramid (`add_pyramid_labels` → `mapping_cluster_category`) and the category-level **temporal-occurrence Wasserstein** ground cost was recomputed. **The finding is confirmed and strengthened at G=28**: the frequency histogram reproduces Patient-vs-Control (0.79 ≈ 0.77) while the *faithful* TW-TWE+DBA barycenter is **near-chance (0.46)**, and **no** method (bigram/ordering, Edit-Shape DTW, soft-mode DBA, re-selected HP) BH-significantly beats majority voting. The paper's **0.84** (RIL-vs-Control) matches the finer **G=77** histogram, not G=28 (0.77). Full record + mechanism in **§12**. **Track A is now closed too (2026-06-05, NB `06e`, §12.7): both decisive ordering-vs-frequency tests are NEGATIVE** — the incremental AUC of bigram ordering over the histogram is ≤0 everywhere, and the faint TBI-vs-RIL Edit-Shape DTW hint collapses to near-chance at full segment-level length. **Verdict: go to Track B (paper reframing).** `pytest` → **85 passed**.

---

## 1. What was broken and is now fixed (NB06)

The notebook did **not run at all** on this server, and even where it ran it computed the wrong distance. Fixes (source-level, committed in the notebooks):

| Bug | Symptom | Fix |
|-----|---------|-----|
| `from aeon.distances._rtwe`/`._twe` imports | Stock aeon 1.2.0 lacks these private modules → `HAS_AEON=False` → **entire notebook skipped** | Re-source rTWE from `smartflat.engine.distances._rtwe`; keep only genuinely-aeon fns behind `HAS_AEON` |
| Hardcoded macOS `SMARTFLAT_DATA_ROOT` (cell 1) | Overrode host auto-detection | Removed; rely on `get_data_root()` |
| `D_twe = twe_pairwise_distance(...)` (cell 10) | **The mission's flagged regression** — standard TWE, drops `D_G`, collapses proposed onto ordinal | `rtwe_pairwise_distance(X, nu, lmbda, precomputed_distances=D_G)` |
| Symbols include noise `-1` | `D_G[symbol]` out-of-range / silent negative-indexing | Remap `X+1` (noise `-1`→background `0`, prototype `s`→`s+1`); hard symbol-range gate |
| `D_G` = full 647×647 K-space | Embedding baselines intractable | Restrict to the 77 occurring symbols (`D_G[:77,:77]`) — **exact** for rTWE |
| `pd` shadowed by a `p_delete` variable (descriptive sweep) | `chapter_6_hyperparameters_searches.png` + `convergence.csv` failed | Renamed local var |

`pytest tests/test_baselines.py` → **45 passed** (39 original + 6 new for the added mode-DBA / p_match functions).

## 2. Experimental setup (as run here)

- **Representation:** `symb_labels` (K-space prototypes). 120 administrations after filtering (HEALTHY 24 / RIL 37 / TBI 59) — note: the manuscript states **26** controls / 122 total; 2 controls are dropped here by the `incomplete_clinical_administrations` / dedup / `notna(symb_labels)` filters.
- **Symbol alphabet:** 76 occurring prototypes + background → **G = 77** (the paper's headline analysis uses a coarser **G = 28** semantic vocabulary; see §3).
- **Sequence length:** resampled (nearest-neighbour, categorical-safe) to a fixed **L = 64**. Chosen so the pure-Python embedding baselines (`dba_dtw`, `soft_dtw`, cost ∝ L^~3.5) stay tractable for the 10×3 protocol; rTWE / TWE-DBA are fast at any L, and discrimination is **insensitive to L** (see §3.2).
- **Hyperparameters (re-selected, §5):** ν = 0, λ = 0.5, δ = 0.3.
- **Protocol:** 10 stratified 50/50 splits × 3 inits; AUC per pairwise comparison + a pooled Patient-vs-Control pass; native distance per method; Wilcoxon + Benjamini–Hochberg.

## 3. The core finding (diagnostic evidence)

### 3.1 The signal is symbol-frequency, not temporal alignment
A plain **symbol-frequency histogram** (76 prototypes, L2 distance to the group-mean histogram, 10×50/50 splits) reproduces the paper almost exactly:

| Comparison | Histogram (here) | Paper §5.4 |
|---|---|---|
| RIL-vs-Control | **0.84** | 0.84 |
| TBI-vs-RIL | **0.57** | 0.59 |
| Patient-vs-Control | **0.80** | 0.77 |

So the representation is fine and the signal is strong — it lives in **which actions occur**, which the histogram (and the Wasserstein-barycenter baseline) capture directly.

### 3.2 aeon's mean-based TWE-DBA is categorically wrong → near-chance
aeon's `elastic_barycenter_average(distance='twe')` **averages nominal prototype indices** (e.g. symbols 2 and 70 → 36), which is meaningless for categorical symbols and erases the frequency signal. The proposed method with this barycenter is **near chance at every length**:

L-sensitivity (proposed = TWE-DBA + {p_match, distance}, paper's ν=1e-5/λ=0.1/δ=0.3, 10 splits):

| L | p_match (RILvsCtrl / TBIvsRIL / PatvsCtrl) | distance |
|---|---|---|
| 64  | 0.53 / 0.53 / 0.54 | 0.56 / 0.55 / 0.51 |
| 128 | 0.60 / 0.51 / 0.59 | 0.53 / 0.47 / 0.55 |
| 256 | 0.51 / 0.47 / 0.49 | 0.48 / 0.48 / 0.51 |
| 512 | 0.51 / 0.52 / 0.51 | 0.52 / 0.52 / 0.49 |

Higher L does **not** help — confirming the issue is the barycenter, not the resampling.

### 3.3 Mode-based DBA (categorical-correct) recovers part of the signal but does not beat the histogram
A **mode-based DBA** (per-aligned-position majority; the paper's described "mode update", added to the harness as `barycenter_mode_dba`) lifts the proposed method to ~0.71 / 0.48 / 0.60 (mode + p_match, L=128) — clearly better than mean-based, but **still below the frequency histogram** (0.84 / 0.57 / 0.80). The proposed method's temporal-alignment machinery does not, on this data, add discriminative value over a simple frequency summary.

**Implication for the paper:** the §5.5 claim that TW-TWE + DBA is the best method is **not supported** on this data/representation as run here. This should be reconciled against the original thesis pipeline (which used the consolidated **G=28** semantic vocabulary — not present as a column in the symbolization dataframe here; the available label columns are all 76–178 fine prototypes).

## 4. FINAL baseline comparison (10 splits × 3 inits)

Source: `baseline_comparison.csv` (+ `baseline_significance.csv`). The proposed method appears as
four documented variants — barycenter (mode vs mean-aeon) × feature (p_match vs rTWE distance);
`tw_twe_pmatch` (mode-DBA + p_match) is the faithful proposed method (the paper's feature).
Best per column is **bold**.

| Method | Patient vs Control | RIL vs Control | TBI vs RIL |
|---|---|---|---|
| **TW-TWE + DBA (mode, p_match)** — proposed | 0.58 ± 0.10 | 0.76 ± 0.08 | 0.54 ± 0.05 |
| TW-TWE + DBA (mode, rTWE distance) | 0.56 ± 0.12 | 0.73 ± 0.06 | 0.56 ± 0.07 |
| TW-TWE + DBA (mean / aeon, rTWE distance) | 0.51 ± 0.13 | 0.48 ± 0.12 | 0.53 ± 0.09 |
| TWE (ordinal cost) — ablation | 0.54 ± 0.08 | 0.57 ± 0.11 | 0.55 ± 0.08 |
| DBA + standard DTW | 0.62 ± 0.11 | 0.69 ± 0.12 | 0.52 ± 0.06 |
| Soft-DTW barycenter | 0.51 ± 0.06 | 0.57 ± 0.09 | 0.52 ± 0.14 |
| Edit-distance median | 0.73 ± 0.06 | 0.74 ± 0.07 | **0.62 ± 0.07** |
| Wasserstein barycenter | 0.72 ± 0.09 | **0.84 ± 0.05** | 0.59 ± 0.05 |
| k-Medoid (TW-TWE) | 0.60 ± 0.08 | 0.65 ± 0.10 | 0.51 ± 0.08 |
| Majority voting | **0.74 ± 0.05** | 0.76 ± 0.07 | 0.50 ± 0.06 |

(Columns: Patient-vs-Control = pooled `CONTROL_vs_PATIENT`; RIL-vs-Control = `HEALTHY_vs_RIL`;
TBI-vs-RIL = `RIL_vs_TBI`. `soft_dtw` runs at a reduced 3×1 budget — `baseline_budget.json`.)

**The proposed method is mid-pack; the frequency baselines win.** Best per comparison: majority-voting
(Patient-vs-Control, 0.74), Wasserstein histogram (RIL-vs-Control, 0.84 — matching the paper's headline),
edit-distance median (TBI-vs-RIL, 0.62). The faithful proposed method (`tw_twe_pmatch`) reaches 0.58 / 0.76 / 0.54.

### Significance (paired Wilcoxon + Benjamini–Hochberg, reference = proposed `tw_twe_pmatch`)
- **Patient-vs-Control:** majority-voting is **significantly better** than the proposed method
  (Δ = −0.16, BH p = 0.035). edit-median (Δ = −0.15, p = 0.07) and Wasserstein (Δ = −0.14, p = 0.12)
  are higher but not BH-significant.
- **RIL-vs-Control:** Wasserstein (0.84) > proposed (0.76), not significant (p = 0.12).
- No comparison shows the proposed method significantly **beating** any baseline.

### Two ablations DO come out clearly in favour of the design
- **Mode vs mean DBA (categorical correctness):** `tw_twe` (mode) ≫ `tw_twe_mean` (aeon mean) —
  RIL-vs-Control 0.73 vs **0.48** (Δ = +0.28, BH p = 0.035, **significant**). Mean-averaging nominal
  symbol indices is demonstrably broken.
- **Wasserstein cost vs ordinal:** `tw_twe` ≫ `twe_ablation` (same mode barycenter) — RIL-vs-Control
  0.73 vs 0.57 (Δ = +0.16, p = 0.053, near-significant). Sanity gate passes: the Wasserstein ground
  cost materially helps.

**Reading:** the *design choices* of TW-TWE (Wasserstein ground cost, categorical/mode update) are each
justified by an ablation — but the *overall method* does not beat simple frequency summaries on this
K-space representation, because the group signal is dominated by symbol frequency (see §3).

## 5. Hyperparameter selection (re-run, §3.3 / §supp-hp)

Replaced the a-priori pick with a **nested, anti-circular grid search** over (ν, λ, δ) optimising held-out group-discrimination AUC (medoid classifier; inner 5-fold CV on a 70% train-val split; 30% outer hold-out reported). Grid: ν∈{0,1e-6,1e-5,1e-4,1e-3}, λ∈{0,1e-3,1e-2,0.05,0.1,0.2,0.5}, δ∈{0,0.1,…,0.5} (210 points).
- **Selected:** ν = 0, λ = 0.5, δ = 0.3.
- **δ = 0.3 confirms the paper.** ν sits on a flat plateau (ν∈[0,1e-4] within 0.0004 — the paper's 1e-5 is equivalent and preserves the strict metric). **λ = 0.5 vs the paper's 0.1**: the AUC criterion prefers a larger edit penalty than the p_match criterion did. _Caveat:_ λ selected at the grid's upper edge — a wider λ sweep is a noted sensitivity.
- Hold-out AUC at selection: pairwise 0.559, pooled 0.629, inner-CV objective 0.647.
- Figures: `chapter_6_hyperparameter_auc_landscape.png` (new), `chapter_6_hyperparameters_searches.png` (refreshed).

## 6. Barycenter stability (Hamming)

`hamming_stability.json` (mean-based DBA, 30 barycenters/group): **median normalized Hamming = 0.97 (IQR 0.95–0.98)** overall and per group.
⚠️ **This does NOT support the manuscript's "majority of positions consistent" sentence and should not be inserted there.** At G=77 with random init, two barycenters agree at ~1/77 of positions by chance, so raw Hamming saturates near 1 regardless of quality — it overstates instability when symbols are fine (close-but-distinct prototypes count as full mismatches). A D_G-aware stability (mean TW-TWE between barycenters) or the coarser G=28 vocabulary would be the appropriate measure for that sentence.

## 7. Artifacts (`$DATA_ROOT/outputs/symbolic_barycenter/`)

`D_twe.npy`, `D_G.npy` (effective ground cost: 77×77, δ-offset), `X_aeon.npy`, `barycenters.pkl`, `split_data.pkl`, `hyperparameters.json`, `hyperparameter_selection.csv`, `hamming_stability.json`, `convergence.csv`, and figures `chapter_6_hyperparameter_auc_landscape.png`, `chapter_6_hyperparameters_searches.png`, `chapter_6_dba_convergence.png`; plus `baseline_comparison.csv`, `baseline_significance.csv`, `baseline_budget.json`, `baseline_*` figures.

## 8. Recommendations for the manuscript

1. **Do not** fill `tab:baselines` with the near-chance proposed-row numbers or replace the §5.4 headline. The honest table here would show the frequency baselines competitive/superior.
2. **Reconcile the representation:** the paper's 0.77/0.84 require the **G=28** semantic vocabulary; reproduce `tab:baselines` on the manuscript machine with that vocabulary (and the mode-based DBA — aeon's mean-based DBA is categorically wrong).
3. **Keep the two clean ablation results** (Wasserstein-cost ≫ ordinal; mode ≫ mean) — these support the design and can be reported.
4. **§7.4 "Fourth"** can be softened: the ordinal-cost ablation now exists (and favours the Wasserstein cost).
5. **Stability sentence:** use a D_G-aware or G=28 measure; the K-space Hamming (0.97) is not informative here.

## 9. What was changed in the manuscript repo (`/home/perochon/paper-chapter-6-barycenters`)

Conservative, **build-safe** changes only — per the "report honestly + flag, do not overwrite §5.5" decision:
- **Four inert LaTeX comment flags** (`% REPRODUCTION FLAG …`) in `main.tex`, adjacent to `tab:baselines`,
  the stability TODO, §3.3 (hyperparameters), and §7.4 "Fourth" — each summarising the finding and pointing
  here. They are `%` comments: they change **nothing** rendered and cannot break compilation.
- **Two figures copied** into `figures/`: `chapter_6_hyperparameter_auc_landscape.png` (new),
  `chapter_6_dba_convergence.png` (new). The authors' existing `chapter_6_hyperparameters_searches.png`
  was **left untouched** (a refreshed K-space version exists in the outputs dir if wanted).
- **No rendered content changed**: the §5.4 headline, the `tab:baselines` cells, the §3.3 selected values,
  and the stability sentence are all **untouched**. `mystyle.sty` is untouched.

**Build could not be verified on this server.** The manuscript needs TeXLive packages absent here
(`bbm`, `algorithm`, …); `tlmgr` is blocked (local TeXLive 2022 vs remote 2026 cross-release) and `apt`
needs root. The edits are comments + additive figures, so they are compilation-safe regardless. On the
build machine: `sudo apt-get install texlive-fonts-extra texlive-science` (or install `bbm`/`algorithm`)
then `latexmk -pdf main.tex`.

## 10. Companion-code changes (smartflat repo)

- `smartflat/features/symbolic_barycenter/baselines.py`: added `barycenter_mode_dba` (categorical mode-based
  DBA), `pmatch_to_barycenter`, `dist_neg_pmatch`. **`pytest tests/test_baselines.py` → 45 passed** (+6 new tests).
- `notebooks/06_barycenter_averaging.ipynb`, `notebooks/06b_barycenter_baselines.ipynb`: fixed + rewired
  (see §1; NB06b registers the proposed variants, ablation, pooled pass, sanity gate, significance).
- Notebooks are **output-stripped** (0 cell outputs); **no `data/` or `outputs/` committed**.
- Known out-of-scope: NB06's hierarchical community-detection cells (end of notebook) have pre-existing
  errors unrelated to the barycenter pipeline; left as-is.

## 11. Recommended next sessions (kickoff prompts)

**Recommendation.** Run the two technical tasks (reproduce-G28 + audit/improve) **together in one fresh
session** — they are deeply coupled (the histogram already reproduces the paper at G=77, so reproducing
G≈28 alone will not vindicate the alignment method; the audit's experiment list *includes* the G≈28 test).
Use a **fresh session** (this one is long/heavy; the findings are fully captured here). Do the
paper-strategy/TODO consolidation in a **second, separate fresh session** (different focus: the paper repo
+ post-thesis audit docs, not the code).

### Kickoff A+B — "Make TW-TWE+DBA work, and beat majority voting" (fresh session)
> ✅ **DONE (2026-06-05) — see §12.** G=28 reproduced + audited; no method beats majority voting; the faithful
> barycenter is near-chance and the signal is frequency at both G=28 and G=77. New code: `vocab.py`, faithful
> `barycenter_mean_rtwe_dba` + experiment methods in `baselines.py`, notebooks `06c`/`06d`, tests (75 passed).
> The original prompt is kept below for provenance.
>
> Read `smartflat/RESULTS_HANDOFF_barycenters.md` and `smartflat/PAPER_BRIDGE.md` first. Context: a faithful
> run found TW-TWE+DBA is mid-pack on the K-space (G=77) symbolic representation; simple frequency baselines
> (Wasserstein histogram, majority voting) match or beat it; the group signal is symbol frequency. Goals:
> (1) **Reproduce the paper's G≈28 result.** Locate the prototype→semantic-category mapping (the A–J clinical
> scheme ≈28 subcategories from the prototype visual annotation — see `smartflat/annotation_smartflat.py`
> `get_annotation_constants`/`order_categories`, `features/symbolization/main_prototypes_annotation.py`,
> and `get_prototypes_mapping`). Map `symb_labels` → the ≈28 categories, rebuild `D_G` on that alphabet,
> and re-run NB06b. Confirm whether the histogram and TW-TWE+DBA both reach ~0.84 on G≈28.
> (2) **Audit the rTWE / TW-TWE+DBA design** (`smartflat/engine/distances/_rtwe.py`, `_eshape_dtw.py`,
> `features/symbolic_barycenter/`) and explain *why* alignment does not add value over frequency here.
> (3) **Propose + run experiments to beat majority voting** — candidates: (a) score by the alignment that
> uses temporal-occurrence structure the histogram discards (transition/ordering features, Edit-Shape DTW
> outer loop), (b) re-select (ν,λ,δ) *for the mode-DBA + p_match feature* (current HP optimised the medoid
> proxy), (c) ShapeDBA / soft-DBA categorical variants, (d) barycenter-length and mean-vs-mode ablations.
> Keep the harness (`evaluate_baselines`) + native-distance design; re-run `pytest tests/test_baselines.py`
> after any `baselines.py` change. **No p-hacking** — report honestly whether alignment can win.

### Kickoff D — "Settle whether ordering ever beats frequency, then close the loop" (fresh session, code)
> ✅ **DONE (2026-06-05) — see §12.7. Both decisive tests NEGATIVE**: ordering never beats frequency and the
> TBI-vs-RIL hint collapses at full length. New code in `baselines.py` (`evaluate_incremental_ordering`,
> bootstrap CIs), notebook `06e`, `pytest` → 85 passed. **Next is Track B / Kickoff C.** Original prompt below.
>
> Read `smartflat/RESULTS_HANDOFF_barycenters.md` §12 first (G=28 reproduction + audit are done; new module
> `smartflat/features/symbolic_barycenter/vocab.py`; notebooks `06c`/`06d`; `pytest tests/test_baselines.py
> tests/test_vocab.py` → 75 passed). Context: at both G=28 and G=77 the proposed TW-TWE+DBA is matched/beaten
> by symbol-frequency baselines; the faithful mean-barycenter is near-chance; the only faint positive is
> Edit-Shape DTW on **TBI-vs-RIL (0.59)**. Goals (Track A in §12.6), in order:
> (1) **Decouple "does ordering help at all" from the barycenter**: train a plain classifier (logistic/RF,
> nested CV) on `[unigram histogram]` vs `[histogram ⊕ bigram transitions]`; report the *incremental* AUC of
> ordering over frequency per comparison. (2) **Pre-registered TBI-vs-RIL follow-up**: Edit-Shape DTW (+ bigram)
> at full length (segment-level / L=median, `step_sequ=1`), 10×N splits, **bootstrap 95% CIs**, a single
> pre-registered hypothesis. (3) Optional: trigram/motif features; a **latent (cosine) category D_G** vs the
> current temporal one; an L=median length-faithful check; frequency-confound controls (length, n_segments,
> background %). Keep the `evaluate_baselines` harness + native-distance design; re-run pytest after any
> `baselines.py` change. **No p-hacking** — a narrow TBI-vs-RIL positive (if any) needs a CI excluding the
> histogram; otherwise report the negative and hand off to Track B (paper reframing).

### Kickoff C — "Consolidate the paper-improvement TODO for a tier-1 venue" (separate fresh session)

_Paste the block below verbatim into a fresh Claude Code session opened in `/home/perochon/paper-chapter-6-barycenters`. It is written to Claude Code best practice: read-before-write, plan-mode gate, testable success criteria, an explicit no-invention guardrail, and a living-document format. Adjust the source filenames if the repo has moved on._

> **Role & goal.** You are doing research synthesis (not coding). Consolidate the scattered audit/review
> notes in this paper repo into a single **living `PAPER_TODO.md`** that drives the chapter-6 barycenter-
> averaging paper to tier-1-venue acceptability. The headline scientific reality is now settled: the proposed
> TW-TWE+DBA does **not** beat symbol-frequency baselines and no ordering signal survives (companion-code
> `RESULTS_HANDOFF_barycenters.md` §12, esp. §12.7) — so the paper's path is **reframing**, not "strengthen the
> method". Hold that as the framing constraint.
>
> **Step 1 — read before writing (do not skip).** Read, in order, then summarise scope back to me in ≤10
> bullets: (a) `PAPER_BRIDGE.md` and `ROADMAP.md` (scope/status), (b) the companion-code findings
> `/home/perochon/smartflat/RESULTS_HANDOFF_barycenters.md` §8 + §12 (the honest results to build on),
> (c) `COMPANION_CODE.md`, `barycenter_corpus/BASELINE_SHORTLIST.md`, `grounding-memory-log.md`, the
> scientific-reviewer output (3.06/5, major revision), any post-thesis audit notes, and an existing
> `PAPER_TODO.md` if present. Use a **subagent** to extract every quantitative claim + its source location so
> your main context stays on structure. If a referenced file is missing, list it and continue — do not invent it.
>
> **Step 2 — plan first (plan mode).** Stay in plan mode: propose the `PAPER_TODO.md` section structure and the
> prioritisation rubric, and get my approval **before** writing the file.
>
> **Step 3 — write `PAPER_TODO.md`.** A living markdown doc with stable `##` section headers as IDs, a per-item
> status marker (`[TODO] / [IN-PROGRESS] / [DONE] / [BLOCKED]`), a one-line owner/effort/impact tag, and a
> dated append-only `## Changelog` at the end. Cover at least: (1) **reframe the contribution** (the central
> decision — principled categorical-sequence averager with ablation-justified design; demote the group-
> discrimination claim) and the concrete `main.tex` edits it implies (§5.4/§5.5, `tab:baselines`); (2)
> **generalisation datasets** where ordering genuinely matters (Breakfast, BPI-Challenge-2012; §7 future-work)
> — the strongest path to a positive headline; (3) **additional baselines/algorithms** (ShapeDBA, GW
> barycenters, block-level edit distances); (4) **reviewer concerns** mapped to specific fixes; (5)
> **submission prioritisation** (effort × impact, blocking dependencies, target venue + deadline).
>
> **Success criteria (verify and show evidence).** `PAPER_TODO.md` exists with the 5 areas above as sections;
> every quantitative claim carries an inline citation to a source (`[RESULTS_HANDOFF §12.7]` / `file:line` /
> reviewer-doc), with **zero** uncited numbers; each item has a status marker and a priority; the changelog has
> today's entry. Verify before declaring done: `grep -c '^- ' PAPER_TODO.md` (expect a substantive list) and
> `grep -nE '[0-9]\.[0-9]{2}|0\.[0-9]+' PAPER_TODO.md` to eyeball that every number near it has a citation.
>
> **Constraints.** No invented results — use only numbers traceable to `RESULTS_HANDOFF`, the notebooks, or the
> reviewer docs; mark anything unverifiable `[NEEDS-SOURCE]` and ask me. Do **not** edit
> `RESULTS_HANDOFF_barycenters.md`, the notebooks, or `main.tex` in this session (TODO only). Stay within the
> barycenter-paper scope (no gaze/kinematics/clinical extensions). Match the existing markdown style.

---

## 12. G=28 reproduction + audit + beat-majority-voting experiments (session 2026-06-05)

Kickoff A+B was run. New code: `smartflat/features/symbolic_barycenter/vocab.py` (G=28 vocabulary +
temporal ground cost), faithful barycenter + experiment methods added to `baselines.py`, notebooks
`06c_g28_reproduction.ipynb` and `06d_audit_and_experiments.ipynb`, tests in `tests/test_vocab.py` +
`tests/test_baselines.py` (**`pytest` → 75 passed**, was 45). Artifacts under
`$DATA_ROOT/outputs/symbolic_barycenter/g28/` (+`/audit`, `/experiments`).

### 12.1 The G=28 vocabulary + D_G (now fully pinned down from the thesis notebooks)
The paper's "G=28" is the **27 named cooking-action categories + 1 background symbol**, built (faithfully,
per `archive-notebooks/demo_rtwe_barycenter_averaging.ipynb`) as:
`filtered_raw_embedding_labels` → `add_pyramid_labels` (→ `pyr_filtered_raw_embedding_labels`) →
`mapping_cluster_category[i][0]` → `update_segmentation_from_embedding_labels(majority_voting_inner,
filter_noise_labels=False)` → integer codes (background=0). Realizes all 27 categories at **76.6 %** token
coverage. (Mapping `symb_labels` directly is wrong — only 19 categories.) `vocab.add_category_columns` does this.
The ground cost is a **28×28 temporal-occurrence Wasserstein-1** matrix (distance = 1-D Wasserstein between
categories' normalized occurrence-time distributions), recomputed via `compute_temporal_distance(
temporal_distance='wasserstein-1')` then the thesis `compute_distance_matrix` transform (+0.3 offset, /max,
background row/col = max). It is **temporal**, not latent/cosine, and was **not** on disk (only K/G-space
variants exist) — `vocab.recompute_category_temporal_D_G` rebuilds it. The thesis barycenter used a **forked
aeon `elastic_barycenter_average`** (mean-petitjean, `precomputed_distances=D_G`) that is **not available**
(stock aeon 1.2.0 lacks it); `baselines.barycenter_mean_rtwe_dba` is a faithful reconstruction (mean +
rTWE alignment; `project='round'/'dg'/'none'`).

### 12.2 Reproduction (cohort HEALTHY 24 / RIL 37 / TBI 59; 10×3 splits; ν=1e-4, λ=0.1)
| Method | RIL-vs-Control | TBI-vs-RIL | Patient-vs-Control |
|---|---|---|---|
| Wasserstein **histogram** (frequency) | **0.77** | 0.53 | **0.79** |
| Majority voting | 0.74 | 0.52 | 0.61 |
| Edit-distance median | 0.75 | 0.49 | 0.72 |
| TW-TWE + DBA — faithful **mean** (`tw_twe_thesis`) | 0.46 | 0.49 | 0.49 |
| TW-TWE + DBA — **mode** | 0.58 | 0.46 | 0.68 |
| TW-TWE + DBA — D_G-Fréchet | 0.58 | 0.51 | 0.53 |

(Paper targets 0.84 / 0.59 / 0.77.) **The histogram reproduces Patient-vs-Control (0.79 ≈ 0.77)**; the
paper's **0.84** RIL-vs-Control matches the **finer G=77** histogram (§3.1), *not* G=28 (0.77) — coarsening
slightly reduces it. **The faithful barycenter is near-chance (0.46)**; mode recovers some signal but stays
below the frequency baselines. **The finding holds at both granularities.**

### 12.3 Audit — why alignment ≈ frequency (06d)
- **p_match is largely a frequency statistic:** Pearson r=0.63 with pure histogram overlap.
- **Alignment is not degenerate:** ~44 % edit moves at λ=0.1 (it *does* warp) — yet **no λ lets the barycenter
  cross the histogram** (0.74 vs best-over-λ 0.66).
- **Mode-collapse:** mode-DBA barycenters have lower symbol entropy (2.8–3.2 bits) than the pooled group
  sequences (3.8) — they drop minority-but-discriminative categories.
- **Mean-vs-mode reconciliation (resolves the §3.2 "mean is broken" caveat):** passing D_G into the *averaging*
  does **not** rescue the index-mean — `mean_rtwe`=0.45 ≈ `aeon_mean`=0.54 (both near chance); `mode`=0.65 /
  `dg_frechet`=0.66 are the categorically-meaningful constructions. So the forked-aeon mean would not have
  helped; the issue is averaging nominal category indices, independent of the alignment cost.

### 12.4 Experiments to beat majority voting — **none succeeded** (honest negative)
Re-selected (ν,λ,δ) for the p_match feature (anti-circular: selected ν=1e-3,λ=0.2,δ=0.3 on disjoint seeds),
plus bigram **transition/ordering** features, **Edit-Shape DTW** outer loop, **soft-mode DBA**. **No method
BH-significantly beats majority voting** (paired Wilcoxon + BH, reference = majority voting).
| Method | RIL-vs-Ctrl | TBI-vs-RIL | Patient-vs-Ctrl |
|---|---|---|---|
| Majority voting | 0.71 | 0.53 | 0.59 |
| Histogram | 0.77 | 0.56 | 0.76 |
| Transition (bigram) | 0.62 | 0.44 | 0.62 |
| Edit-Shape DTW | 0.53 | **0.59** | – |
| Soft-mode DBA | 0.63 | 0.52 | 0.59 |
| TW-TWE mode (re-selected HP) | 0.65 | 0.52 | 0.60 |

**One hint, not significant:** Edit-Shape DTW is best on **TBI-vs-RIL (0.59)** — the comparison where
frequency is weakest (histogram 0.56) — suggesting temporal-ordering carries marginal signal there. Worth a
larger-budget follow-up (full L, more splits) but **does not** currently beat frequency.

### 12.5 Recommendation
Reproduction does not vindicate alignment over frequency at G=28 (or G=77). Reframe the contribution per §8
(the method is a principled categorical-sequence averager whose *design choices* — Wasserstein cost, mode
update — are each ablation-justified, but whose *group-discrimination* does not beat a frequency summary on
this cohort). The only avenue with a faint positive signal is **temporal-ordering on TBI-vs-RIL** (Edit-Shape
DTW) — a targeted, adequately-powered experiment there is the honest next step, not a headline claim.

### 12.6 Recommended next steps (two tracks)

**Track A — improve / continue the experiments (code-side, this repo).** Goal: settle, honestly, whether
*any* temporal/ordering signal beats frequency, and close the remaining loose ends.
1. ✅ **DONE (2026-06-05, NB `06e`) — NEGATIVE. Decouple "does ordering help at all" from the barycenter.**
   Plain classifier (logistic / RF, nested 10×5 CV) on `[unigram histogram]` vs `[histogram ⊕ bigram
   transition features]`; incremental AUC of ordering per comparison. **Ordering never helps** — see §12.7.
2. ✅ **DONE (2026-06-05, NB `06e`) — NEGATIVE. Targeted, pre-registered TBI-vs-RIL follow-up.** Edit-Shape
   DTW (+ bigram) at full length (segment-level / L=median=181, `step_sequ=1`), 10 splits, bootstrap 95% CIs,
   single pre-registered hypothesis. The 0.59 hint **collapses to near-chance** at full length — see §12.7.
3. **Higher-order / motif features.** Trigram profiles, frequent-subsequence (motif) counts, transition-matrix
   spectral features — test against the histogram on TBI-vs-RIL specifically.
4. **Latent (semantic) ground cost instead of temporal.** Current `D_G` is *temporal-occurrence* (correlated
   with frequency/timing). Build a category-level **cosine/latent** `D_G` (aggregate
   `compute_multimodal_matrices('K_space')` to the 28 categories) and re-run — gives the alignment a cost that
   is *not* frequency-correlated; check whether it changes the verdict.
5. **Length-faithful check.** Re-run the key methods at **L=median (~5162) / segment-level** (not L=64) to
   close the resampling question end-to-end (handoff says L-insensitive; this makes it airtight). Budget:
   only the fast methods + rTWE (embedding baselines are intractable at that L).
6. **Frequency-confound controls.** Verify the group signal is category *frequency* and not a proxy
   (sequence length, n_segments, background proportion, fragmentation). Partial out these covariates and
   re-test the histogram.

**Track B — paper track (the manuscript / `paper-chapter-6-barycenters`).**
1. **Reframe the contribution** (the central decision): present TW-TWE+DBA as a *principled categorical-
   sequence averaging* method (Wasserstein ground cost + categorical mode update, each ablation-justified) and
   demote/qualify the group-discrimination claim — the honest `tab:baselines` shows frequency baselines
   competitive/superior at both G=28 and G=77. Do **not** fill `tab:baselines` with the near-chance barycenter row.
2. **Generalisation datasets** (the method may win where ordering genuinely matters): Breakfast,
   BPI-Challenge-2012 (§7 future-work). This is the strongest path to a positive headline.
3. **Additional baselines/algorithms**: ShapeDBA, GW barycenters, block-level edit distances.
4. **Keep the clean ablations** (mode ≫ mean; Wasserstein ≫ ordinal) — they support the *design* and can be
   reported as-is.

**Decision for the user:** Track A item 1 (incremental-AUC test) + item 2 (TBI-vs-RIL follow-up) are quick and
decisive about whether *any* alignment/ordering claim survives — recommended **before** committing to a paper
reframing. If the answer is "no" (likely), proceed straight to Track B reframing + new datasets.
> ✅ **RESOLVED (2026-06-05, §12.7): the answer is "no".** Both decisive tests are negative — ordering never
> beats frequency, and the faint TBI-vs-RIL hint does not survive at full length. **Proceed to Track B**
> (paper reframing + generalisation datasets). Track A items 3–6 are not warranted by these results.

### 12.7 Track A — decisive ordering-vs-frequency tests (session 2026-06-05, NB `06e`) — **both NEGATIVE**

New code (`baselines.py`): `histogram_features`, `transition_features`, `evaluate_incremental_ordering`
(nested-CV logistic+RF, paired across feature sets, bootstrap-CI + Wilcoxon on the incremental Δ),
`bootstrap_auc_ci`, `bootstrap_delta_ci`. Notebook `06e_ordering_vs_frequency.ipynb` (pre-registration
stated up front). Tests added → **`pytest` 85 passed** (was 75). Artifacts under
`$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/` (`incremental_ordering*.csv`, `tbi_ril_followup*`).

**(1) Incremental AUC of bigram ordering over the unigram histogram** (paired Δ = `both − hist`):

| Comparison | clf | AUC hist | AUC both | Δ | Δ 95% CI |
|---|---|---|---|---|---|
| CONTROL_vs_PATIENT | logreg / rf | 0.70 / 0.75 | 0.56 / 0.68 | −0.15 / −0.08 | both **< 0** |
| HEALTHY_vs_RIL | logreg / rf | 0.80 / 0.83 | 0.65 / 0.83 | −0.15 / **0.00** | < 0 / **brackets 0** |
| RIL_vs_TBI | logreg / rf | 0.61 / 0.59 | 0.54 / 0.51 | −0.07 / −0.08 | both **< 0** |

Adding bigram ordering **never improves** held-out AUC — the incremental-Δ 95% CI is ≤0 in every
comparison×classifier (best case: no effect; for linear models the 784-dim bigram block overfits and *hurts*).
Ordering carries no group signal beyond symbol frequency.

**(2) Pre-registered TBI-vs-RIL follow-up** (Edit-Shape DTW, segment-level L=median=181, `step_sequ=1`, 10 splits):
- histogram AUC **0.64** [0.60, 0.68]; **Edit-Shape DTW 0.52** [0.49, 0.56]; transition 0.56 (secondary).
- Paired Δ(eshape − histogram) = **−0.117**, 95% CI **[−0.184, −0.050]** — entirely below 0.
- **Verdict: NEGATIVE** (`delta_ci_low` < 0). The L=64 "0.59 hint" does not reproduce at full segment-level
  length; Edit-Shape DTW falls to near-chance while the histogram holds at 0.64.

**Conclusion.** No temporal/ordering signal beats symbol frequency at any granularity (G=28, G=77) or length
(L=64, segment-level). Track A is exhausted. **Go to Track B** (reframe the contribution; pursue
generalisation datasets where ordering genuinely matters — the strongest path to a positive headline).
