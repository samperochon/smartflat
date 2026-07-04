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

---

## 13. Session 2 (2026-06-29) — faithful embedding-level rep + full-length (L≈5162) confirmation

**Branch `barycenter-faithful-s2`** (off `main`; faithful 06c already merged to `main` as `82f56b8`).
Earlier `06*` runs resampled every sequence to **L=64**, which *crippled* the alignment methods.
Session 2 moved 06/06d/06e to the **embedding-level** representation and **hardened the rTWE kernel**
so the **definitive full-length test is tractable**.

### 13.1 Kernel hardening (so L≈5162 is runnable)
`smartflat/engine/distances/_rtwe.py`: `_rtwe_distance_rolling` (2-row buffer, O(L) memory,
**213 MB → 83 KB/call**, bit-identical to the full cost matrix) + flat-`prange` pairwise. Full pairwise
**n=122, L=5162 ≈ 3 min** (was intractable). The full matrix is kept only for alignment *paths*.
The historical large-L "deadlock" was **OMP + a forked kernel**, not L → run notebooks with
`NUMBA_THREADING_LAYER=workqueue` (baked into each first cell). Tests: **37 pass** in
`tests/test_distances_rtwe.py` + `tests/test_barycenter_dba.py`.

### 13.2 Headline — the 0.84 is the G=77 histogram, confirmed at full length (NB `06`)
Native-distance AUC, 10×{2–3} splits, swept over resample length L (mean AUC):

| Comparison | method | L=64 | 256 | 1024 | **5162** |
|---|---|---|---|---|---|
| **Control vs RIL** | `wasserstein` (freq.) | 0.84 | 0.83 | 0.82 | **0.82** |
| | `k_medoid` (align.) | 0.65 | 0.72 | 0.77 | **0.79** |
| | `tw_twe_mode` (align. bary.) | 0.77 | 0.76 | 0.78 | — |
| | `majority_voting` | 0.76 | 0.80 | 0.82 | 0.82 |
| **TBI vs RIL** | `wasserstein` | 0.59 | 0.62 | 0.61 | **0.61** |
| | `k_medoid` | 0.51 | 0.54 | 0.57 | **0.59** |
| **Patient vs Control** | `wasserstein` | 0.72 | 0.77 | 0.75 | **0.75** |
| | `k_medoid` | 0.60 | 0.57 | 0.69 | **0.70** |
| | `majority_voting` | 0.74 | 0.76 | 0.80 | 0.80 |

- The **frequency histogram is length-invariant** (~0.82–0.84 Control-vs-RIL = the paper's **0.84**) and
  **leads at every L**.
- The alignment exemplar `k_medoid` was crippled at L=64 (0.65) and **rises monotonically with L**
  (→ 0.79 at L≈5162) but **plateaus just below** the histogram — it **never overtakes frequency**, even
  at the true embedding length. So the L=64 "near-chance" was a resampling artifact, *and* the faithful
  result is a clean plateau below frequency.

**G=28 (NB `06c`) — same picture at the consolidated vocabulary.** Mean AUC, same 10×{1–3}-split harness
and L sweep; at the full length the mode-DBA `tw_twe_mode` is gated off (O(L²)/member), so only the
pairwise-cheap methods run (`n_inits=1`, lossless as they are deterministic):

| Comparison | method | L=64 | 1024 | **5162** |
|---|---|---|---|---|
| **Control vs RIL** | `wasserstein` (freq.) | 0.77 | 0.80 | **0.80** |
| | `k_medoid` (align.) | 0.54 | 0.70 | **0.71** |
| | `tw_twe_mode` (align. bary.) | 0.53 | 0.68 | — |
| **Patient vs Control** | `wasserstein` | 0.78 | 0.80 | **0.79** |
| | `k_medoid` | 0.61 | 0.71 | **0.70** |
| | `tw_twe_mode` | 0.60 | 0.63 | — |

- Same verdict as G=77: the **histogram is L-invariant (~0.80) and leads at every L**; `k_medoid` rose
  through the mid-L range and **plateaus (~0.70–0.71) just below** it at the full embedding length, never
  overtaking frequency. This **closes the open follow-up** stated in 06c's own verdict (full pairwise
  n=120/L=5162 ≈ 3 min; cached row appended to `g28/auc_vs_L_faithful.csv`).

### 13.3 Audit holds at faithful L=512 (NB `06d`)
p_match ≈ frequency-overlap (Pearson **r=0.61**, ≈ the 0.63 at L=64); **no λ lets the barycenter+p_match
cross the histogram** (0.78 vs 0.75 best-over-λ); mean-DBA is broken (D_G-aware `mean_rtwe` **0.41** vs
`mode` 0.62 / `dg_frechet` 0.56; the stock-aeon TWE-DBA does not even run at L=512). In the beat-MV table
the **only** BH-significant win over majority voting is the **frequency histogram** (Patient-vs-Control,
0.77 vs 0.63, BH p≈0.01); **no alignment/ordering method beats MV**.

### 13.4 Ordering still loses on the faithful (segment-level) rep (NB `06e`)
Re-run of §12.7 on the artifact-free segment-level features: Goal 1 `ordering_helps=False` for all 6
(comparison × classifier); Goal 2 pre-registered eshape **NEGATIVE** (eshape 0.53 vs histogram 0.64,
Δ=−0.11, 95% CI excludes 0). Confirms §12.7 was not an L=64 artifact.

### 13.5 Artifacts & how the paper repo ingests them
**Committed (in-repo, readable by anyone with the repo):**
- The executed notebooks **with figures embedded** — `notebooks/06_barycenter_averaging.ipynb`,
  `06d_audit_and_experiments.ipynb`, `06e_ordering_vs_frequency.ipynb` (commit `145ddde`). These are the
  primary, self-contained record (tables in cell outputs + 10/6/1 embedded figures respectively).
- This handoff (§13) + the `PAPER_BRIDGE.md` 2026-06-29 update — the prose/number summary.

**Local only (gitignored under `$DATA_ROOT/outputs/symbolic_barycenter/`; regenerate by running the
notebooks):** machine-readable tables and standalone PNGs —
`auc_vs_L_kspace.csv` + `auc_vs_L_kspace.png`, `chapter_6_hyperparameters_searches.png`,
`chapter_6_dba_convergence.png`, `chapter_6_non_bg.png`; and under `g28/audit/` + `g28/experiments/`:
`beat_mv_comparison_g28.csv`, `beat_mv_significance_g28.csv`, `incremental_ordering_summary.csv`,
`tbi_ril_followup_ci.json`, plus the audit/experiment PNGs.

**To feed the paper repo** (`paper-chapter-6-barycenters/`): copy the needed `chapter_6_*.png` /
`*_auc*.png` from `$DATA_ROOT/outputs/symbolic_barycenter/` into `paper-chapter-6-barycenters/figures/`
and cite the numbers from the §13.2 table (or the `*.csv` files), recording provenance in the paper's
`COMPANION_CODE.md` (per the §5 coordination protocol). The figures are *also* embedded in the committed
notebooks, so the notebook is a sufficient fallback source if the data dir is unavailable.

---

## 14. Next-session kickoffs (post full-length confirmation, 2026-06-30)

With the L≈5162 plateau confirmed at G=28 (§13.2), four follow-up directions are scoped as **separate fresh
sessions**. Full paste-able prompts (Claude-Code best-practice: read-before-write, plan-mode gate, testable
success criteria, no-p-hacking, reusable-code-in-package) live as standalone files in `.claude/prompts/`
(**local-only — `.claude/` is gitignored**, like the existing `session-6b-kickoff.md`; move them to a tracked
path if they should travel with the repo).
Honest framing for all of them: these **strengthen the method and the rigor / test new hypotheses** — none is
"tune the barycenter to beat the frequency histogram on SDS2" (that signal is settled absent; §12–§13).

| Kickoff | File | Goal (one line) | Success target |
|---|---|---|---|
| **E** | `.claude/prompts/kickoff-E-order-evaluation.md` | Rigorous, reusable evaluation of how much group-discriminative info lives in symbol *order* vs frequency, anchored by a **frequency-preserving order-shuffle null** (the cheapest, highest-leverage rigor move). | A CI'd `ΔAUC(order)` per comparison + `order_evaluation.py` + tests. |
| **F** | `.claude/prompts/kickoff-F-barycenter-methods.md` | **Multi-session arc.** Compare categorical-sequence barycenters (current TW-TWE+mode-DBA vs **FGW**, soft-DTW/SSG, profile-HMM/MSA) applied from their *original papers*, judged on **group-representativeness/fidelity** (NOT discrimination AUC). FGW's α gives a principled frequency↔structure decomposition. | Approved arc plan + `barycenter_quality.py` harness + FGW integrated/tested + methods×quality table. |
| **G** | `.claude/prompts/kickoff-G-behavioral-metrics.md` | A reusable, well-tested **behavioral-structure metrics** library (perseveration, transition entropy, dwell-time, fragmentation, complexity) — *local* disorganisation that frequency and *global* ordering both miss; clinically motivated for dysexecutive syndrome. | `structure_metrics.py` in-package + tests + honest CI'd group comparison. |

**Deferred — D: controlled/synthetic ordering-sensitive validation dataset.** A dataset where *order*, not
*frequency*, is discriminative would let the method demonstrate it recovers order signal *when present*,
converting the negative SDS2 result into a clean validity proof (operationalises Track B). **Deferred** — it
needs design thought (how to generate it: permutation classes, motif insertion, or a process-model generator;
how to match frequency across classes; what "order signal" to plant). Mirror this note into the paper repo's
`ROADMAP.md` (`paper-chapter-6-barycenters/`) as future-work before scheduling a session.

---

## 15. Order evaluation — frequency-preserving shuffle-null ΔAUC (Kickoff E, 2026-06-30)

**Branch `barycenter-order-eval`** (session E of the coordinated E→G→F arc; roadmap
`.claude/plans/coordination-EFG-roadmap.md`). The decisive **higher-leverage** ordering probe the prior
tests lacked. §12.7/§13.4 measured ordering with an *incremental* bigram block on top of the histogram —
which overfits (`G²=784` features vs n≈60–120) and conflates "does order help" with "does this
representation help". §15 replaces it with a **frequency-preserving order-shuffle null**: score the best
order-aware classifier on the *intact* sequences and on many *within-sequence* shuffles that preserve each
sequence's symbol multiset **exactly**, and report **ΔAUC = AUC_intact − mean(AUC_shuffled)** with a 95% CI
from the shuffle distribution. Because the shuffle holds frequency fixed and the **same pipeline** scores
intact and shuffled data, classifier optimism cancels in ΔAUC — the null is its own control.

**Two nulls** (both per-sequence ⇒ the unigram histogram is exactly invariant): `token` (uniform
permutation; destroys order **and** dwell → ΔAUC = all structure beyond bare frequency) and `runlength`
(permute the order of runs; preserves per-symbol dwell-time distribution → ΔAUC = *pure sequencing*, frequency
**and** dwell held). They coincide at segment-level (runs length-1) and separate at embedding-level.

**Verdict: NEGATIVE across the board — 0 / 15 (comparison × feature × null) cells show an order signal**
(pre-registered rule: order present iff ΔAUC 95% CI `ci_low > 0`). Native nested-CV (logreg, `RepeatedStratifiedKFold`
2×5), `n_shuffles=100`, on the faithful G=28 cohort (HEALTHY 24 / RIL 37 / TBI 59).

| rep | feature | null | comparison | AUC_intact | ΔAUC | 95% CI | p_perm |
|---|---|---|---|---|---|---|---|
| segment | transition | token | CONTROL_vs_PATIENT | 0.72 | +0.080 | [−0.043, +0.209] | 0.14 |
| segment | transition | token | HEALTHY_vs_RIL | 0.78 | +0.039 | [−0.062, +0.148] | 0.28 |
| segment | run_transition | token | RIL_vs_TBI | 0.68 | +0.045 | [−0.054, +0.141] | 0.20 |
| embedding | transition | token | RIL_vs_TBI | 0.63 | **−0.073** | **[−0.132, −0.009]** | 0.98 |
| embedding | transition | runlength | HEALTHY_vs_RIL | 0.84 | +0.096 | [−0.004, +0.221] | 0.05 |
| embedding | transition | runlength | CONTROL_vs_PATIENT | 0.71 | +0.054 | [−0.058, +0.158] | 0.23 |
| embedding | run_transition | runlength | RIL_vs_TBI | 0.66 | +0.042 | [−0.049, +0.143] | 0.22 |

(Full 15-row table: `$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/order_shuffle_null_deltaAUC.csv`;
ΔAUC-with-CI bar chart `…/order_shuffle_null_deltaAUC.png`, also embedded in NB `06f`.)

- **Every CI brackets 0.** The two extremes only *strengthen* the negative: embedding `transition×token`
  on **RIL-vs-TBI** has ΔAUC **−0.073, CI entirely below 0** — intact order *actively hurts* (the
  dwell-dominated frame-bigram overfits the hardest comparison, exactly the §12.7 failure mode the null now
  exposes cleanly). The single faint *hint* is embedding `transition×runlength` on **HEALTHY-vs-RIL**
  (ΔAUC +0.096, CI [−0.004, +0.221], p=0.050) — lower bound −0.004, so **not** a signal under the
  pre-registered rule; at most a candidate for a powered follow-up, not a result.
- **Probe validated by construction** (`tests/test_order_evaluation.py`): a planted-order synthetic (identical
  frequencies, different transition grammar) gives ΔAUC CI **> 0** (positive control), and a frequency-only
  synthetic gives ΔAUC CI **bracketing 0** even at AUC_intact=1.0 (negative control — the null does not
  hallucinate order from frequency). So the SDS2 negative is a real absence, not an underpowered probe.

**Implication (consistent with §12–§14, now via the strongest available probe).** *The group-discriminative
signal is in **which** actions occur and **how often**, not **in what order**.* This hardens the paper's
central honest claim; it does **not** change `tab:baselines`. F reads this to set expectations (structure-
awareness should matter little for discrimination on SDS2); the order question is settled absent here, and a
positive demonstration needs the deferred synthetic ordering-sensitive dataset (§14, Deferred-D).

### Shipped (reusable, in-package)

- `smartflat/features/symbolic_barycenter/order_evaluation.py` — `token_shuffle`, `runlength_shuffle`,
  `order_shuffle_null`, `run_transition_features` (dwell-invariant run-grammar bigrams), `order_information`
  (the CI'd ΔAUC evaluator). `tests/test_order_evaluation.py` (18 tests incl. both controls). NB
  `notebooks/06f_order_shuffle_null.ipynb`.
- **Arc shared symbols (E owns; G & F import — pin to these):**
  - `baselines._make_clf(name)` and `baselines._nested_cv_auc(F, y, splits, pipe, grid)` — extracted from
    `evaluate_incremental_ordering` (refactored onto them; **behavior-identical, `test_baselines.py` 73/73
    unchanged**). G's `evaluate_incremental_structure` and E's `order_information` both call the single
    `_nested_cv_auc`.
  - `vocab.load_g28_cohort(rep='int_cat_segm_embedding_labels', out_dir=None, upsample_to=None, …)` →
    `(df, X_symbolic, labels)` and `vocab.build_g28_ground_cost(out_dir=None, offset_value=0.3,
    method='max_rows_cols_pre')` → `D_G_cat (28,28)` — factored from 06c Cells 2–3.
  - Full suite **91 passed**.
- **Divergences from the roadmap contract (flagged):** (1) branched off the current HEAD
  (`barycenter-faithful-s2`), **not `main`** — `main` was 4 commits behind and lacked §13/§14, so §15 could
  not append "after §14" off it; the arc's serial-merge intent is preserved (my branch contains the
  prerequisite state). (2) `load_g28_cohort` returns `X_symbolic` as a **ragged list** by default (the
  embedding-level rep is genuinely ragged, L 2292–9570, so a bare `vstack` is impossible); pass
  `upsample_to=L` for a rectangular `(n, L)` matrix. Per-sequence order/structure features need no rectangular
  form, so this is strictly more faithful (no resample artifact). Everything else matches the contract.

---

## 16. Behavioral-structure metrics — local disorganisation / perseveration (Kickoff G, 2026-07-01)

**Branch `barycenter-structure-metrics`** (off E's tip `78d7ef4`). New module
`smartflat/features/symbolic_barycenter/structure_metrics.py` + `tests/test_structure_metrics.py`
(23 tests; full suite **187 passed** — the 2 `test_dataset.py` errors are pre-existing/unrelated).
Reuses E's shared helpers (`_make_clf`, `_nested_cv_auc`, `load_g28_cohort`, `_rle`) — **no** re-copied
CV loop, cohort cells, or RLE.

**Motivation.** E (§15) settled that *global* symbol order carries no group signal beyond frequency
(0/15 ΔAUC). G tests an **orthogonal** hypothesis: *local* execution structure — fragmentation,
perseveration, complexity, temporal drift — that the frequency histogram **and** global-order tests both
miss. The module is a first-class, vocabulary-agnostic, NaN-safe library of **per-sequence** metrics
(transition entropy, immediate-repeat/switch rate, run-length & dwell stats, fragmentation index, n-gram
coverage, normalised Lempel–Ziv, background fraction, first-vs-second-half drift), each a small
unit-tested function, plus `compute_structure_metrics` (aggregator), `structure_features` (matrix),
`structure_group_stats` (Cliff's δ + bootstrap CI + Hedges g + **BH across the full 48-test family**),
and `evaluate_incremental_structure` (leakage-guarded nested CV; calls `_nested_cv_auc`).

### 16.1 The result is a genuine POSITIVE — local structure adds signal beyond frequency

Leakage-guarded incremental AUC, 10×5 RepeatedStratifiedKFold; struct = 13 de-collinearised metrics:

| Comparison | clf | AUC hist | AUC struct-only | AUC both | Δ(both−hist) | Δ 95% CI |
|---|---|---|---|---|---|
| **CONTROL_vs_PATIENT** | logreg / rf | 0.74 / 0.74 | **0.81 / 0.81** | 0.81 / 0.82 | **+0.069 / +0.077** | **[0.045,0.093] / [0.044,0.111]** |
| HEALTHY_vs_RIL | logreg / rf | 0.85 / 0.88 | 0.83 / 0.83 | 0.90 / 0.89 | +0.051 / +0.018 | [0.017,0.085] / [−0.012,0.048] |
| RIL_vs_TBI | logreg / rf | 0.70 / 0.73 | 0.65 / 0.61 | 0.72 / 0.73 | +0.020 / −0.002 | [0.001,0.039] / [−0.022,0.017] |

Patient-vs-Control: structure alone reaches 0.81 (vs hist 0.74) and the incremental Δ over frequency
excludes 0 for **both** classifiers — the honest positive E's global-order test did not find. HEALTHY-vs-RIL
is partial (logreg only, hist already strong); RIL-vs-TBI null.

### 16.2 Descriptive per-metric group stats (Cliff's δ, BH over 48 tests) — with a length caveat

24 / 48 (metric × comparison) tests survive BH, **but sequence length is a group-correlated confound**
(RIL median L≈6215 > HEALTHY≈4330), and several metrics scale with L. Splitting the BH-significant metrics
by their Spearman-with-length `r_L`:

- **Length-robust (|r_L| < 0.3) — the genuine local-structure signals:** `fragmentation_index`
  (Ctrl-vs-Pat δ=+0.48, r_L=0.23 — **patients more fragmented**), `halves_tv_distance` /
  `halves_js_divergence` (δ≈−0.48 / −0.42, r_L≈−0.05 — **patients drift *less*** between task halves: more
  uniform, less phase-structured execution), `transition_entropy` (HEALTHY-vs-RIL δ=+0.37),
  `trigram_coverage` (δ=+0.44). `background_fraction` also separates (δ=+0.34) but is a frequency
  restatement (histogram bin 0), not structure.
- **Length-driven (|r_L| ≥ 0.5) — read with caution:** `immediate_repeat_rate` / `switch_rate` (|r_L|=0.88),
  `run_length_mean` (0.88), `run_length_max` (0.53), `dwell_mean_over_states` (0.73). Their separation
  largely reflects RIL's longer administrations, not perseveration per se.
- `lz76_complexity`, `run_length_cv`, `dwell_cv_over_states`, `bigram_coverage` separate no group.

### 16.3 Length control — the Patient-vs-Control gain survives; HEALTHY-vs-RIL does not

Length **alone** separates groups (Cliff's δ: HEALTHY-vs-RIL **+0.72**, Ctrl-vs-Pat **+0.49**,
RIL-vs-TBI **−0.40**) and adds held-out AUC over frequency (e.g. Ctrl-vs-Pat 0.74→0.77–0.79; HEALTHY-vs-RIL
0.88→0.94 rf) — a real confound, not just a metric-level correlation. The decisive test — shipped as the
recallable `evaluate_structure_length_controlled` — is the incremental AUC of struct over `[hist ⊕ length]`
(structure beyond frequency **and** duration):

| Comparison | clf | Δ(struct \| hist+len) | 95% CI | verdict |
|---|---|---|---|---|
| **CONTROL_vs_PATIENT** | logreg / rf | **+0.038 / +0.033** | **[0.014,0.061] / [0.008,0.059]** | **survives — both clf** |
| HEALTHY_vs_RIL | logreg / rf | +0.006 / −0.032 | [−0.019,0.031] / [−0.048,−0.017] | does **not** survive (was length) |
| RIL_vs_TBI | logreg / rf | −0.001 / −0.016 | [−0.015,0.013] / [−0.034,0.002] | null |

**The Patient-vs-Control structure signal is real beyond both frequency and length** (Δ +0.033–0.038, CI
excludes 0, both classifiers). The apparent **HEALTHY-vs-RIL** structure gain is a **duration effect** —
it vanishes (logreg) or reverses (rf) once length is controlled. **RIL-vs-TBI** is null throughout.

### 16.4 Verdict & API

The deliverable is the **library**; the honest scientific finding is a **length-robust positive for local
execution structure on Patient-vs-Control** — a clinically-motivated axis (patients are more *fragmented*,
drift *less* across task phases, and differ in transition entropy) that neither the frequency histogram nor
E's global-order test captures — while HEALTHY-vs-RIL reduces to sequence duration and RIL-vs-TBI is null.
This is orthogonal to §15 (global order) and complements the joint narrative: *frequency dominates global
discrimination (E); local execution structure is a separate, real axis on Patient-vs-Control (G).*

**Public API** (recall from any session):
`from smartflat.features.symbolic_barycenter.structure_metrics import (compute_structure_metrics,
structure_features, structure_group_stats, evaluate_incremental_structure,
evaluate_structure_length_controlled)` — plus the individual per-sequence metric functions (all
vocabulary-agnostic, NaN-safe; `on={'embedding','segment'}` where a representation choice exists).
**Honest-use notes baked into the API:** metric-level group tests should
always be read against the `spearman_length` column `structure_group_stats` emits (length is a
group-correlated covariate); the predictive claim is the length-controlled one in §16.3. Artifacts under
`$DATA_ROOT/outputs/symbolic_barycenter/g28/structure/` (`structure_metrics_table.csv`,
`structure_group_stats.csv`, `incremental_structure_summary.csv`, `length_control_summary.csv`).

---

## 17. Barycenter-quality methods comparison — FGW + representation metrics (Kickoff F, 2026-07-02)

**Branch `barycenter-quality-fgw`** (session **F** of the E→G→F arc; roadmap
`.claude/plans/coordination-EFG-roadmap.md`, branched off G's tip). This arc **reframes the evaluation
axis**: the group-*discrimination* question is settled (§12–§16; E's §15 = 0/15 order signals), so F asks
which averaging method is the best **group representative** — judged on *representation fidelity*, not AUC.
Shipped this session: the FGW barycenter + a method-agnostic quality harness + the first methods×metric
table and α-decomposition. Soft-DTW/SSG/ShapeDBA/MSA are deferred to later arc sessions (per plan).

### 17.1 Method + harness (reusable, in-package)

- **`baselines.barycenter_fgw`** — Fused Gromov–Wasserstein barycenter (Vayer, Chapel, Flamary, Tavenard,
  Courty, *ICML 2019* / *Algorithms 2020*; POT `ot.gromov.fgw_barycenters`). Each sequence is a graph
  (nodes = timesteps, features = per-symbol vector, structure `C` = normalized `|i−j|`). POT weights **α**
  on the **GW structure** term, so **α→0 = feature/frequency (Wasserstein)**, **α→1 = structure-only (GW)**.
  Two node encodings: **`fgw_mds`** (classical-MDS of `D_G`, so α→0 ≈ the project's
  `barycenter_wasserstein(D_G)`) and **`fgw_onehot`** (one-hot, D_G-agnostic). Continuous FGW centroids are
  decoded to hard symbols by nearest prototype. Registry helper `baselines.fgw_methods` (same
  `{build, distance}` contract, native rTWE distance). N=128 nodes; O(N²)/iter — length-gated.
- **`barycenter_quality.score_barycenter_quality`** — scores any `{build, distance, kind}` registry method
  identically, auto-dispatching over output kinds (sequence / histogram / medoid-index / transition-matrix).
  Metrics: rTWE within-group inertia (common yardstick), native inertia, frequency fidelity
  (Wasserstein-on-`D_G`), temporal-structure preservation (bigram Frobenius), symbol entropy + distinct +
  segment counts, and init-stability. `quality_table` pivots to methods×metric; `build_fgw_registry` emits
  a full α-sweep. Tests: `tests/test_barycenter_quality.py` (11, incl. FGW shape/determinism/α-knob and the
  harness NaN-pattern/medoid/stability contracts). E/G frozen suites unchanged (**152 passed**).

### 17.2 Methods × quality at G=28 (L=128; HEALTHY 24 / RIL 37 / TBI 59)

Length-gated (mirrors 06c): the O(L²·G) embedding-space DTW baselines `dba_dtw` / `soft_dtw` (~130 s/group
at L=128) are **excluded here**, deferred to a shorter-L / later run. Mean over groups:

| method | freq_fidelity ↓ | entropy (bits) | rTWE inertia ↓ | struct_pres ↓ | n_distinct | n_segments |
|---|---|---|---|---|---|---|
| wasserstein (histogram) | **0.027** | 3.73 | — | — | 28 | — |
| **fgw_onehot** | **0.038** | 3.55 | 76.3 | 3.29 | 18 | 113 |
| **fgw_mds** | **0.043** | 3.50 | 78.8 | 3.27 | 17 | 114 |
| edit_median | 0.090 | 3.54 | 46.0 | 2.59 | 17 | 58 |
| shape_dba | 0.105 | 3.00 | 44.6 | 2.62 | 13 | 59 |
| k_medoid | 0.120 | 3.35 | 46.5 | 2.71 | 16 | 54 |
| **tw_twe_mode** (paper) | 0.120 | 3.02 | **44.8** | 2.66 | 13 | 57 |
| majority_voting | 0.193 | 2.45 | **44.2** | 2.55 | 9 | 43 |
| transition | — | — | — | ~0 (is the mean) | — | — |

(group-sequence entropy reference ≈ 3.7 bits. `tw_twe_mode` and `shape_dba` are composed inline in `06g`
via `barycenter_mode_dba` / `barycenter_soft_mode_dba`, not standalone `baselines` registry functions.
Artifacts: `$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/barycenter_quality_methods_L128.csv`;
per-method chronograms `chronogram_*.png`; notebook `06g_methods_chronogram.ipynb`.)

- **FGW is the most frequency-faithful, least mode-collapsed structured averager**: it nearly matches the
  `wasserstein` histogram on frequency fidelity (0.038–0.043 vs 0.027) and entropy (≈ 3.5 vs group ≈ 3.7),
  while the paper's **`tw_twe_mode` collapses** (entropy 3.02, fidelity 0.120, only 13/28 symbols) — as does
  `majority_voting` (2.45, 0.193, 9/28), the §12.3 mode-collapse made quantitative.
- **The alignment methods win the rTWE yardstick** (`tw_twe_mode`/`majority_voting` ≈ 44 vs FGW ≈ 78): they
  are the most *compact* summaries in the warped geometry they optimise. FGW optimises its own objective, not
  rTWE — **an honest tradeoff, not a single winner**: which method is "best" depends on the chosen fidelity
  axis. FGW is init-stable (rTWE-inertia std ≈ 1.3–1.7 of ≈ 78; histogram std ≈ 1e-3).

### 17.3 FGW α-decomposition (the frequency↔structure knob)

Full α∈{≈0, .25, .5, .75, 1} sweep, both encodings (**no cherry-picked α**):

| α | freq_fidelity ↓ (mds / onehot) | entropy (mds / onehot) |
|---|---|---|
| ≈0 | 0.040 / 0.037 | 3.59 / 3.62 |
| 0.5 | 0.043 / 0.037 | 3.52 / 3.54 |
| 0.75 | 0.045 / 0.036 | 3.53 / 3.53 |
| **1.0** | **0.259 / 0.190** | **2.97 / 2.47** |

- For **α ≤ 0.75** FGW stays frequency-faithful (≈ 0.04) and entropy near the group level (≈ 3.5), and
  **α→0 approaches the `wasserstein` histogram** — the nested special case confirmed empirically (POT
  documents `0<α<1`; the α≈0 anchor validates it). At **α = 1** (structure-only GW) both **degrade sharply**.
- **Adding pure temporal structure does not improve the group representative — it hurts it.** This is exactly
  what E's §15 predicts (structure carries little content beyond frequency): **the α knob is understood, not
  tuned to win.** Figure `fgw_alpha_decomposition.png`; table `barycenter_quality_fgw_alpha.csv`.

### 17.4 Verdict & API

The deliverable is the **library + reframe**: the barycenter is a *principled, ablation-justified averager*
whose structure knob is characterised, not a frequency-beater. On representation, **FGW is the strongest
frequency-faithful / anti-collapse structured average**, while the paper's mode-DBA is the most rTWE-compact
— reported as an honest per-axis tradeoff. Consistent with the joint narrative: *frequency dominates (E);
local execution structure is a separate real axis on Patient-vs-Control (G); the averager's frequency↔
structure knob is understood and does not need to win (F).*

**Public API** (recall from any session): `from smartflat.features.symbolic_barycenter.barycenter_quality
import (score_barycenter_quality, quality_table, build_fgw_registry)` and
`from smartflat.features.symbolic_barycenter.baselines import (barycenter_fgw, fgw_methods)`. Consumes E's
shared loaders (`vocab.load_g28_cohort`, `vocab.build_g28_ground_cost`); touches `baselines.py` only to add
FGW (CV helpers untouched). **Deferred (later arc sessions):** Soft-DTW / SSG subgradient, ShapeDBA
(off-label for categorical), profile-HMM / progressive-MSA positional consensus, and `dba_dtw`/`soft_dtw` at
a tractable length.

---

## 18. Library-backed Soft-DTW + SSG barycenters into the quality harness (Kickoff F·S2, 2026-07-02)

**Branch `barycenter-quality-fgw`** (session **F·S2**, continuing F on the same branch). Adds the two
averagers §17 deferred — now **library-backed (tslearn 0.8.1), not hand-rolled** — and scores them with the
**existing** §17 harness (extend-by-reuse; **no harness change**). Method-development + **gated preview**
session: implement, unit-test, and print a G=28 preview at L=128; the full-length (L≈5162) / full-cohort run
is a **`pomme` hand-off** (stub in `06i`, not run locally).

### 18.1 Methods (reusable, in-package)

- **`baselines.barycenter_softdtw`** — Soft-DTW barycenter (Cuturi & Blondel, *ICML 2017*) via
  `tslearn.barycenters.softdtw_barycenter` (L-BFGS-B) on the `D_G` embedding
  (`embed_symbolic_to_real`), decoded to hard symbols (`project_real_to_symbolic`). `gamma→0` → DTW.
  **Deterministic** (Euclidean-mean init; `random_state` accepted for the registry contract but unused).
- **`baselines.barycenter_ssg`** — Stochastic-subgradient DTW averaging (SSG; Schultz & Jain, *Pattern
  Recognition 2018*) via `tslearn.barycenters.dtw_barycenter_averaging_subgradient`; same embed→decode
  adapter; **reproducible under a fixed `random_state`**.
- **`baselines.softdtw_ssg_methods`** — `{build, distance}` registry (mirrors `fgw_methods`): `soft_dtw_bary`
  scored by its **native soft-DTW** distance (`dist_soft_dtw`), `ssg` by the **native rTWE** distance
  (`dist_rtwe`, like `fgw_*`/`k_medoid`). Both output symbol sequences ⇒ the harness auto-computes every axis.
  tslearn is a **lazy optional import** (install-hint on `ImportError`, matching the `aeon`/`ot` convention).
- Tests: `tests/test_barycenter_softdtw_ssg.py` (6, `pytest.importorskip('tslearn')`: shape/valid-symbol/
  determinism for both, SSG seed-sensitivity, embedding round-trip, harness integration + `quality_table`
  pivot). `pytest tests/test_barycenter_softdtw_ssg.py tests/test_barycenter_quality.py -q` → **17 passed**;
  E/G frozen suites (`test_order_evaluation`/`test_structure_metrics`/`test_baselines`/`test_barycenter_dba`)
  → **117 passed**, unchanged.

### 18.2 Extended methods × quality at G=28 (L=128, **gated preview budget**; HEALTHY 24 / RIL 37 / TBI 59)

Reduced budget (`soft_dtw_bary`/`ssg` `max_iter=10`, `n_inits=2`). **Not the scale run** — full length /
full cohort deferred to pomme. tslearn is compiled, so at L=128 these run ~2–5 s/group (vs §17's ~130 s/group
pure-numpy `soft_dtw`/`dba_dtw`, still omitted). Mean over groups (group-sequence entropy ref ≈ 3.78 bits):

| method | freq_fidelity ↓ | entropy (bits) | rTWE inertia ↓ | struct_pres ↓ | n_distinct | n_segments |
|---|---|---|---|---|---|---|
| wasserstein (histogram) | **0.027** | 3.73 | — | — | 28 | — |
| fgw_onehot | 0.037 | 3.55 | 75.8 | 3.28 | 18 | 111 |
| fgw_mds | 0.044 | 3.50 | 77.8 | 3.27 | 17 | 113 |
| **ssg** | 0.138 | 3.24 | **47.1** | 2.94 | 16 | 65 |
| **soft_dtw_bary** | 0.165 | 3.17 | **48.7** | 2.93 | 14 | 26 |
| tw_twe_mode (paper) | 0.120 | 3.02 | **44.8** | 2.66 | 13 | 57 |

(Artifacts: `$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/barycenter_quality_softdtw_ssg_L128.csv`;
per-method chronograms `chronogram_{soft_dtw_bary,ssg,fgw_onehot,tw_twe_mode}.png`; notebook
`06i_softdtw_ssg_quality.ipynb`.)

- **Both new methods land on the §17 tradeoff frontier — no new winner.** As DTW-family alignment averagers
  they are **rTWE-compact** (≈ 47–49, near the paper's `tw_twe_mode` ≈ 45 and far tighter than FGW ≈ 76–78),
  but they trade that off against **frequency fidelity** (0.14–0.17 vs the histogram's 0.027 and FGW's
  0.037–0.044) and **entropy** (partial mode-collapse: 3.2–3.3 vs group ≈ 3.78). They sit *between* FGW
  (frequency-faithful, anti-collapse) and `tw_twe_mode` (most rTWE-compact, most collapsed) — two more
  principled points on the same frontier, consistent with §17's "best depends on the chosen fidelity axis."
- **`soft_dtw_bary` collapses more than `ssg`** (14 distinct / 26 segments / entropy 3.17 vs 16 / 65 / 3.24):
  the soft-DTW centroid is smoother, so decoding yields longer runs of few symbols. Its `inertia_native` is
  the soft-DTW *discrepancy* (negative, −54, as expected for the γ-regularised cost) — not comparable across
  methods; use the common `inertia_rtwe` yardstick.
- **Stability** (`n_inits=3`): `soft_dtw_bary` deterministic (rTWE-inertia std = 0.000); `ssg` std ≈ 0.86 on
  ≈ 47 (≈ 1.8% — modestly init-stable).

### 18.3 Verdict & API

Two more **principled, library-backed** averagers now score through the same harness; they confirm — rather
than overturn — §17's honest per-axis tradeoff (rTWE-compact but less frequency-faithful, with `soft_dtw_bary`
the more collapse-prone). No cherry-picked winner. Consistent with the joint narrative: *frequency dominates
discrimination (E); local structure is a separate real axis on Patient-vs-Control (G); the averagers are a
characterised family on a frequency↔structure–vs–warp-compactness frontier, not frequency-beaters (F).*

**Public API** (recall from any session): `from smartflat.features.symbolic_barycenter.baselines import
(barycenter_softdtw, barycenter_ssg, softdtw_ssg_methods)`, scored via the unchanged
`barycenter_quality.{score_barycenter_quality, quality_table}`. Consumes E's shared loaders
(`vocab.load_g28_cohort`, `vocab.build_g28_ground_cost`); touches `baselines.py` only to add the two builders
+ registry (CV helpers / FGW untouched). tslearn is an optional dependency (lazy import).
**Pomme hand-off:** `06i` final cell documents the full-length (L≈5162) / full-cohort / full-budget
invocation, clearly marked NOT run locally. **Deferred (later sessions):** ShapeDBA (off-label for
categorical), profile-HMM / progressive-MSA positional consensus (S3), and the pomme scale run.

### 18.4 Fairness note — how each averager treats the discrete symbol space

Raised during F·S2 review: is every benchmarked averager developed with full respect to the **discrete
(categorical)** nature of the symbolic state sequences? **No — by design there are two families**, and the
paper should state the mechanism split explicitly (full reference, with mechanism analysis and the decode
caveats, in **[`BARYCENTER_METHOD_DISCRETENESS.md`](BARYCENTER_METHOD_DISCRETENESS.md)**).

- **Family A — native-categorical** (never leave symbol space; output positions are genuine symbols chosen by
  voting/median/medoid): `tw_twe_mode` (`barycenter_mode_dba`, the paper method — rTWE-align + per-position
  **hard majority vote**, re-discretised **every iteration**), `shape_dba` (soft `D_G`-vote → argmax),
  `majority_voting`, `edit_median` (alphabet substitutions), `k_medoid` (a real member sequence), plus the
  order-free `wasserstein` (symbol histogram) and `transition` (bigram matrix).
- **Family B — continuous relaxation + decode** (embed symbol → its `D_G` row via `embed_symbolic_to_real`,
  average real vectors, then snap to nearest `D_G` row via `project_real_to_symbolic` **only at the end**):
  `dba_dtw`, `soft_dtw`, **`soft_dtw_bary`** (new), **`ssg`** (new), and `fgw_*` (decode to nearest prototype).

**Three asymmetries** (Family B does *not* respect discreteness during optimisation): (1) the mean of symbol
embeddings is generally between symbols — the "mean of category 2 and 70 → 36" pathology mode-DBA was built to
avoid; (2) DBA-family drifts continuously and snaps **once**, whereas mode-DBA re-projects to a valid symbol
**every** iteration; (3) the decode is **Euclidean** in `D_G`-row space, not the `D_G`/Wasserstein-optimal
symbol. **Benchmark is still fair**: the harness scores every sequence-emitting method on the same rTWE-inertia
yardstick (on decoded symbols) *and* reports each on its native distance — the §17 "honest tradeoff" framing.
**No over-claim:** collapse is not unique to Family B (`tw_twe_mode` is the *most* collapsed, 13 distinct /
3.02 bits) — the two families collapse by different mechanisms (dominant-symbol voting vs snap-of-smoothed-mean).

**Current handling note (fairness):** `tw_twe_mode` is reimplemented **natively** here (vendored
`rtwe_alignment_path`), **not** via a forked aeon — no hidden fork asymmetry vs the stock-tslearn `ssg` /
`soft_dtw_bary`. **Deferred (fresh session, `.claude/prompts/kickoff-K-discreteness-fairness.md`):** Lever 2 =
`D_G`-argmin decode (ground-cost-consistent snap, reported as an Euclidean-vs-`D_G` ablation); Lever 3 =
per-iteration re-discretisation for `dba_dtw`/`ssg` (a categorical variant reported *alongside* the continuous
ones). Both additive, ablation-reported, harness-reused, tests-first.

---

## 19. MSA positional-consensus barycenter into the quality harness (Kickoff F·S3, 2026-07-02)

**Branch `barycenter-quality-fgw`** (session **F·S3**, the **last method in the F roster**, continuing F on
the same branch). Adds **one** principled positional-consensus averager — the **Family-A native-categorical**
counterpart to F·S1's FGW and F·S2's Soft-DTW/SSG (all **Family B**) — and scores it with the **existing** §17
harness (extend-by-reuse; **no harness change**). Method-development + **gated preview** session: implement,
unit-test, and print a G=28 preview at L=128; the full-length (L≈5162) / full-cohort run is a **`pomme`
hand-off** (stub in `06j`, not run locally). No new dependency — reuses only the vendored rTWE aligner.

### 19.1 Method (reusable, in-package)

- **`baselines.barycenter_msa_consensus`** — **center-star MSA + profile per-column consensus**, entirely in
  symbol space (**no `D_G` embedding, no decode** — the defining Family-A property). Algorithm: (1) pick the
  star **center** = within-group rTWE **medoid**; (2) align every member pairwise to the center via the
  vendored `rtwe_alignment_path` (the same aligner as the paper's `tw_twe_mode`); (3) **merge** the
  pairwise-to-center alignments into one multiple alignment with insertion columns ("once a gap, always a
  gap"); (4) **profile consensus** — Laplace-`pseudocount` argmax per column (mode is the `pseudocount→0`
  limit); (5) **occupancy rule** — keep only columns whose non-gap occupancy ≥ `occupancy` (0.5 match-state
  rule), yielding a gap-free symbol sequence. **Deterministic** (medoid center + argmax; `random_state`
  accepted for the registry contract but unused). Cites Gusfield 1993 (center-star, 2-approximation);
  Feng–Doolittle 1987 / ClustalW (Thompson 1994); Durbin, Eddy, Krogh & Mitchison 1998 (profile HMM,
  pseudocounts, occupancy match-state rule) — all verified via `literature-grounder`.
- **`baselines.msa_consensus_methods`** — `{build, distance}` registry (mirrors `softdtw_ssg_methods`):
  `msa_consensus` scored by the **native rTWE** distance (`dist_rtwe`, like `ssg`/`fgw_*`/`k_medoid`). Output
  is a genuine symbol sequence ⇒ the harness auto-computes every axis.
- Tests: `tests/test_barycenter_msa_consensus.py` (7, **no `importorskip`** — no new dep): shape/valid-symbol/
  stays-in-`[0,G)`, determinism, **consensus of identical sequences == that sequence**, ragged
  insertion/deletion handling, `pseudocount→0` recovers hard mode, harness integration + `quality_table`
  pivot. `pytest tests/test_barycenter_msa_consensus.py tests/test_barycenter_softdtw_ssg.py
  tests/test_barycenter_quality.py tests/test_order_evaluation.py tests/test_structure_metrics.py
  tests/test_baselines.py -q` → **139 passed** (E/F/G frozen suites unchanged).

### 19.2 Extended methods × quality at G=28 (L=128, **gated preview budget**; HEALTHY 24 / RIL 37 / TBI 59)

Deterministic build (`n_inits=2` for the shared table). **Not the scale run** — full length / full cohort
deferred to `pomme`. Group-sequence entropy reference ≈ **3.78 bits**. `inertia_rtwe` ↓ = more compact;
`freq_fidelity` (Wasserstein-on-`D_G`) ↓ = better; `struct_preservation` (bigram Frobenius) ↓ = better.

| method | family | inertia_rtwe | freq_fidelity | struct_pres | entropy_bits | n_distinct | n_segments |
|---|---|---|---|---|---|---|---|
| **`msa_consensus`** | **A** | **44.36** | 0.114 | 2.597 | 3.142 | 13.7 | 52.7 |
| `tw_twe_mode` | A | 44.79 | 0.120 | 2.661 | 3.022 | 13.3 | 56.7 |
| `ssg` | B | 47.14 | 0.138 | 2.940 | 3.237 | 16.0 | 65.0 |
| `soft_dtw_bary` | B | 48.72 | 0.165 | 2.929 | 3.165 | 13.7 | 26.3 |
| `fgw_onehot` | B | 75.79 | 0.037 | 3.279 | 3.545 | 18.0 | 111.2 |
| `fgw_mds` | B | 77.81 | 0.044 | 3.268 | 3.504 | 17.5 | 113.2 |
| `wasserstein` | A (freq) | — | 0.027 | — | 3.729 | 28.0 | — |

Init-stability: `msa_consensus` is **deterministic** → `stability_inertia_rtwe = 0.0` exactly.

### 19.3 Honest per-axis reading (no cherry-picked winner)

`msa_consensus` lands right next to its precedent `tw_twe_mode` — as expected for a Family-A alignment voter —
**marginally more rTWE-compact** (44.36 vs 44.79) and **slightly less collapsed** (3.14 vs 3.02 bits, i.e. the
occupancy-gated insertion columns retain a touch more symbol diversity), with similar frequency fidelity. It is
**not** a new leaderboard winner: `wasserstein` still owns frequency fidelity (0.027) and the anti-collapse
`fgw_*` still keep the most entropy (~3.5 bits, closest to the 3.78-bit reference). The point is
**representativeness, not a race**: `msa_consensus` is the **statistically-principled Family-A positional
consensus** (a proper MSA with insertion columns + a profile/occupancy rule), offered as the native-categorical
alternative to the ad-hoc lock-step `majority_voting` and the iterative single-reference `tw_twe_mode`. Fairness
frame unchanged from §18.4: common rTWE yardstick + each method's native distance; both families reported.

### 19.4 Artifacts + `pomme` hand-off

- Notebook **`06j_msa_consensus_quality.ipynb`** (committed with figures) — registry merge with the §17/§18
  anchors, extended methods×quality table with the `family` column, deterministic-stability check, 4 chronograms
  (`msa_consensus`, `tw_twe_mode`, `ssg`, `fgw_onehot`), and the **`pomme` scale stub (not run locally)**.
- Outputs: `$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/barycenter_quality_msa_consensus_L128.csv`
  + `chronogram_msa_*.png`.
- **Deferred to `pomme`** (`# --- run at scale on pomme ---` in `06j`): the full-length (L≈5162, no upsample)
  / full-cohort `score_barycenter_quality` run. Dominant cost is the O(n²·L²) pairwise-rTWE **medoid**
  selection (~1600× the L=128 per-cell cost). The **optional profile-HMM decode** (`hmmlearn.CategoricalHMM`)
  + any new dependency were **explicitly deferred** — not justified for the core positional consensus.
- `BARYCENTER_METHOD_DISCRETENESS.md` Family-A table gains the `msa_consensus` row.

---

## 20. Discreteness-fairness levers — ground-cost decode + categorical variants (Kickoff K, 2026-07-03)

**Branch `barycenter-quality-fgw`** (session **K**, continuing F on the same branch, off F·S3's tip `5feadc3`).
A **fairness follow-up** to F·S2: the Family-A/B split (§18.4, `BARYCENTER_METHOD_DISCRETENESS.md`) documented
that Family B optimises in a continuous relaxation and imposes discreteness only by a final **Euclidean**
decode. This session ships the two remaining **algorithmic** levers that make Family B's treatment of
discreteness match the geometry the harness scores in — **additively, ablation-reported, no cherry-picked
winner** (existing keys/defaults byte-for-byte unchanged; §17/§18/§19 reproduce; sibling suites **139 passed**,
unchanged). Method-development + **gated preview** (G=28, L=128); the full-length (L≈5162) / full-cohort run
is a **`pomme` hand-off** (stub in `06k`).

### 20.1 Levers + API (reusable, in-package)

- **Lever 2 — ground-cost-consistent decode knob.** `project_real_to_symbolic(..., decode={'euclidean','dg'})`.
  A Family-B barycenter position is a real vector `m` in the `D_G`-row embedding. rTWE (the yardstick) uses a
  **direct `D_G[a,b]` substitution cost**, so for a DBA mean `m = mean_i D_G[s_i]` the ground-cost-optimal
  symbol is `argmin_c m[c] = argmin_c mean_i D_G[c,s_i]` — the **vocabulary-restricted 1-medoid under `D_G`**
  (exact for DBA means; a profile-argmin heuristic for the optimised soft-DTW/SSG centroids). Default
  `'euclidean'` = unchanged; lossless round-trip for pure symbols under both decodes. Threaded through
  `barycenter_{dba_dtw,softdtw,ssg,fgw}`. For **`fgw_onehot`** (feature term is D_G-agnostic, no `D_G`-row
  embedding) the `'dg'` decode is the **barycentric ground-cost projection** `argmin_c (D_G @ p)[c]` with
  `p` = clip⁺+L1-normalised centroid (the only researcher DOF; stated in the docstring). `feature='mds'`+`'dg'`
  **raises** (MDS feature cost already ≈ `D_G`, so a separate variant would be degenerate).
- **Lever 3 — per-iteration re-discretised categorical variants.** `barycenter_dba_dtw(discretise_each_iter=True)`
  (snap+re-embed inside the loop), and new `barycenter_ssg_cat` / `barycenter_softdtw_cat` (outer loop over
  tslearn with a small inner `max_iter`, snapping between rounds — both accept an init array). Keeps the
  reference a **valid symbol string every iteration** (like mode-DBA, but **mean-then-snap** rather than a hard
  vote; precedent cited: `barycenter_mode_dba`). Each per-iter snap uses the `'dg'` ground-cost medoid. The
  atomic step is `_snap_and_reembed`.
- **Registry** `baselines.discreteness_lever_methods(D_G, ...)` — **7** additive `{build, distance}` entries
  beside the existing methods: `dba_dtw_dg` / `soft_dtw_bary_dg` / `ssg_dg` / `fgw_onehot_dg` (Lever 2);
  `dba_dtw_cat` / `ssg_cat` / `soft_dtw_bary_cat` (Lever 3). Each native distance mirrors its base method
  (`dist_rtwe` for DBA/SSG/FGW, `dist_soft_dtw` for the soft-DTW pair) ⇒ the unchanged §17 harness scores every
  axis. Same `random_state`/`n_inits` as the Euclidean sibling ⇒ a `*_dg` variant shares its sibling's
  continuous barycenter and differs **only** in the decode (clean ablation).
- Tests: `tests/test_barycenter_discreteness_levers.py` (**19**, `pytest.importorskip` for tslearn/ot):
  decode-knob shapes/determinism/round-trip + the worked counterexample (dg→1, euclidean→0), the
  `_snap_and_reembed` invariant, categorical-stays-symbolic (horizon sweep + embed-input spy), FGW dg + the
  MDS `ValueError`, and harness/registry integration. `pytest tests/test_barycenter_discreteness_levers.py
  tests/test_barycenter_softdtw_ssg.py tests/test_barycenter_msa_consensus.py tests/test_barycenter_quality.py
  tests/test_order_evaluation.py tests/test_structure_metrics.py tests/test_baselines.py -q` → **19 + 139**
  passed (E/F/G frozen suites unchanged).

### 20.2 Ablation A — decode: Euclidean vs `D_G` (G=28, L=128, gated; both ways)

Same continuous barycenter per pair (identical seeds); only the decode differs. `inertia_rtwe` ↓ = more
compact; `freq_fidelity` ↓ = better. Anchors: `wasserstein` freq 0.027 / entropy 3.73; `tw_twe_mode` inertia
44.79 / entropy 3.02; group-sequence entropy reference ≈ **3.78 bits**.

| method | decode | inertia_rtwe ↓ | freq_fidelity ↓ | struct_pres ↓ | entropy_bits | n_distinct | n_segments |
|---|---|---|---|---|---|---|---|
| `dba_dtw` | euclidean | 45.65 | 0.114 | 2.73 | 3.23 | 16.0 | 60.2 |
| `dba_dtw` | **`D_G`** | **44.77** | 0.119 | 2.65 | 3.07 | 13.8 | 59.0 |
| `soft_dtw_bary` | euclidean | 48.72 | 0.165 | 2.93 | 3.17 | 13.7 | 26.3 |
| `soft_dtw_bary` | **`D_G`** | **46.16** | 0.164 | 2.55 | 2.99 | 11.0 | 22.0 |
| `ssg` | euclidean | 47.14 | 0.138 | 2.94 | 3.24 | 16.0 | 65.0 |
| `ssg` | **`D_G`** | **45.98** | 0.147 | 2.82 | 3.03 | 13.3 | 61.7 |
| `fgw_onehot` | euclidean | 75.79 | 0.037 | 3.28 | 3.55 | 18.0 | 111.2 |
| `fgw_onehot` | **`D_G`** | 76.20 | 0.041 | 3.36 | 3.57 | 18.7 | 112.0 |

### 20.3 Ablation B — continuous (snap@end) vs categorical (per-iteration)

| method | variant | inertia_rtwe ↓ | freq_fidelity ↓ | struct_pres ↓ | entropy_bits | n_distinct | n_segments |
|---|---|---|---|---|---|---|---|
| `dba_dtw` | continuous | 45.65 | 0.114 | 2.73 | 3.23 | 16.0 | 60.2 |
| `dba_dtw` | **categorical** | 45.28 | 0.122 | 2.62 | 3.09 | 14.0 | 56.3 |
| `ssg` | continuous | 47.14 | 0.138 | 2.94 | 3.24 | 16.0 | 65.0 |
| `ssg` | **categorical** | 45.45 | 0.145 | 2.69 | 3.05 | 13.3 | 60.0 |
| `soft_dtw_bary` | continuous | 48.72 | 0.165 | 2.93 | 3.17 | 13.7 | 26.3 |
| `soft_dtw_bary` | **categorical** | 46.19 | 0.166 | 2.49 | 2.98 | 11.0 | 23.3 |

### 20.4 Honest per-axis reading (no cherry-picked winner)

- **Both levers move the DTW-family embed→decode methods toward the `tw_twe_mode` corner** — more
  **rTWE-compact** (its design goal: it minimises the very ground cost the yardstick measures; inertia drops
  ≈ 0.9–2.6 across `dba_dtw`/`soft_dtw_bary`/`ssg`) at the cost of being **slightly more collapsed** (entropy
  −0.1 to −0.2, fewer distinct symbols) with **frequency fidelity roughly unchanged** (±0.01). So treating
  discreteness more faithfully makes Family B behave more like Family A's hard voters — it does **not** turn
  them into frequency-faithful averagers. Neither decode/variant dominates across axes: `D_G`/categorical win
  rTWE-inertia (by construction); Euclidean/continuous keep marginally more entropy/diversity.
- **`fgw_onehot` is the exception — the barycentric `D_G` decode is essentially a wash** (inertia 75.8 → 76.2,
  freq 0.037 → 0.041, entropy 3.55 → 3.57, slightly *more* distinct), exactly as expected for a method whose
  feature term is D_G-agnostic: the ground-cost projection only reshuffles the snap at the margin. It stays the
  anti-collapse, frequency-faithful structured average of §17.
- **Verdict:** the fairness levers are now available and characterised, but the §17–§19 story is **unchanged** —
  representativeness is an honest per-axis tradeoff, not a race. The value is *auditability*: Family B's decode
  can now be made ground-cost-consistent and its optimisation kept categorical throughout, and the ablation
  shows exactly what that costs (a shift toward compact-but-collapsed), reported both ways per axis.

### 20.5 Artifacts + `pomme` hand-off

- Notebook **`06k_discreteness_levers_quality.ipynb`** (committed with figures) — anchors + lever registry, the
  full methods×quality table with the `family` column, the two ablation tables (with per-axis Δ), 3 chronograms
  (`dba_dtw` under Euclidean / `D_G` / categorical), and the **`pomme` scale stub (not run locally)**.
- Outputs:
  `$DATA_ROOT/outputs/symbolic_barycenter/g28/experiments/barycenter_quality_discreteness_levers_L128.csv`
  + `chronogram_lever_*.png`.
- **Deferred to `pomme`** (`# --- run at scale on pomme (NOT run locally) ---` in `06k`): the full-length
  (L≈5162, no upsample) / full-cohort / full-budget `score_barycenter_quality` run. Dominant cost is the
  O(L²) pure-numpy `dba_dtw` build (gated to `dba_max_iters=6` in the preview).

---

## 21. Arc audit — E→F→G maintainability/accuracy/rigor/reusability/consistency (Kickoff L, 2026-07-03)

**Branch `barycenter-arc-audit-L`** (off `barycenter-quality-fgw` tip `ba43e62`). A read-heavy
expert audit of everything the E→F→G→(H/I/J/K) arc added, across five dimensions, plus a phased
behavior-preserving refactoring plan. **Full report: [`ARC_AUDIT.md`](ARC_AUDIT.md)** (severity-ranked,
`file:line`-cited, per-dimension verdicts). Frozen suites confirmed **209 passed** at session start.

**Headline:** the arc is scientifically sound and unusually well-documented — every method is
faithful to its cited source (`literature-grounder` 5/5 SUPPORTED: FGW↔POT/Vayer, Soft-DTW↔Cuturi,
SSG↔Schultz-Jain, MSA↔Gusfield/Durbin, rTWE↔Marteau TWE), the statistics are leakage-guarded and
honestly reported, and the registry/harness is genuinely reusable. **No P0 survived verification.**
The real debt is structural/consistency: **(P1)** `baselines.py` = 2109 lines doing ≥6 jobs, and
**(P1)** the `nu`/`lmbda` default drifted into **four** operating points (`0.001/1.0` vs `1e-4/0.1`
vs the eshape inner `0.001/0.01` vs the doc's `1e-5/0.1`) — committed numbers are safe (notebooks
pin `1e-4/0.1`) but the mismatched defaults are a latent trap.

**Phase 0 landed this session (behavior-preserving):** extracted the copy-pasted `_ground_cost`/
`_cohort`/`_seq` test helpers to `tests/_bary_helpers.py` (7 files); canonical
`barycenter_quality.FAMILY`/`family_of()` + `tests/test_family_taxonomy.py` (de-duplicates the
06i/06j/06k literal for future notebooks); `visualization.py` stale-TODO + misplaced-import
cleanup; `PAPER_BRIDGE.md` Tier-D reconciliation (the §17–§20 quality roster is *additive
representativeness*, not `tab:baselines` discrimination); this §21 + `ARC_AUDIT.md`. **Deferred to
phased sub-sessions** (`.claude/plans/kickoff-l-squishy-meadow.md`): Phase 1 `nu`/`lmbda`
unification behind `RTWE_NU`/`RTWE_LMBDA` constants gated by a `quality_table` CSV **oracle no-op
proof**; Phase 2 `baselines.py` split (distances/builders/registries + back-compat re-exports);
Phase 3 reusability polish (`__init__` re-exports, `rtwe_alignment_path` arg-order footgun); Phase 4
rigor caveats (fold-level-bootstrap anti-conservatism, length-standardization notes).

---

## 22. rTWE `nu`/`lmbda` unification behind module constants (Kickoff M, 2026-07-03)

**Branch `barycenter-nu-lmbda-unify`** (off the arc-audit tip `24c531a`; executes `ARC_AUDIT.md` **Phase 1**).
A pure **internal-consistency, behaviour-preserving** session — no new method, no new result. The arc's rTWE
cost default had drifted into four operating points (`ARC_AUDIT.md` Dimension 5); this collapses the
**smartflat layer** to one canonical point behind module constants, gated by a numeric no-op proof so every
committed §17–§20 number stays byte-for-byte reproducible.

- **Constants + single source of truth.** `baselines.RTWE_NU = 1e-4`, `baselines.RTWE_LMBDA = 0.1` are now
  the default for **all 16** smartflat-layer functions/registries carrying a `nu`/`lmbda` default —
  `dist_rtwe`, `pmatch_to_barycenter`, `dist_neg_pmatch`, `barycenter_mode_dba`, `default_baseline_methods`
  (the **five** that had drifted to `0.001/1.0`), plus the six `*_methods` registry builders,
  `barycenter_msa_consensus`/`barycenter_mean_rtwe_dba`/`barycenter_soft_mode_dba`, the `*_stock_twe` pair,
  and `dist_eshape_dtw` (already `1e-4/0.1`, now referencing the constant) — and `barycenter_quality`'s
  common yardstick (`score_barycenter_quality`, `build_fgw_registry`, imported from `baselines`).
- **Vendored kernel untouched.** `engine/distances/_rtwe.py`'s aeon-style defaults (`0.001/1.0`) stay —
  smartflat callers always pass explicit values. The eshape inner-cost hardcode (`0.001/0.01`) is named
  `_ESHAPE_INNER_RTWE_NU/LMBDA` and documented as a fixed, near-inert, deliberately-decoupled inner cost
  (**byte-identical**, not threaded — threading would move `test_distances_eshape_dtw`).
- **Naming.** `barycenter_dba_dtw(max_iters=…)` → `max_iter=` with a deprecation-warning back-compat
  `max_iters` alias; the 2 internal + test callers migrated. (`barycenter_soft_dtw` `decode=` parity
  deferred to Phase 3.) `PAPER_BRIDGE.md` §6 footnoted: thesis grid `1e-5/0.1` vs shipped `1e-4/0.1`
  (genuinely different; not reconciled in code).
- **Oracle no-op proof (the gate).** On the clean checkout vs post-edit, the **L=128 merged 5-registry**
  (`default + fgw + softdtw_ssg + msa + discreteness`, 18 methods, all built with explicit `1e-4/0.1`)
  `quality_table` and a gated `evaluate_baselines` were frozen to CSV. **Both byte-identical before/after**
  (md5 unchanged: quality_table `6b79866…`, evaluate `05c511c…`; empty `diff`). The gated override even
  exercised the `max_iters=` alias, still byte-identical. **No migration surfaced** — the repo-wide
  call-site audit found zero caller relying on the old default for a `nu`/`lmbda`-sensitive value (every
  default-reliant site asserts invariants — self-distance/symmetry/non-negativity/key-membership — or uses
  `majority_voting`, which is `nu`/`lmbda`-independent). The committed §17–§20 numbers are preserved exactly.
- **Guard test.** `tests/test_rtwe_hparam_consistency.py` (21 cases): `inspect.signature` over all 16
  functions asserts each `nu`/`lmbda` default `== RTWE_NU/RTWE_LMBDA` (none is `0.001`/`1.0`); bare
  `dist_rtwe` `==` explicit-canonical `==` pinned `tests/_oracles/rtwe_hparam_oracle.csv`; `max_iter` alias
  equivalence + `DeprecationWarning`. Frozen suite (11 prior files + the guard) → **234 passed, 13 warnings**
  (was 213; +21 guard cases, and the warning count is unchanged — no stray deprecation warnings leaked from
  the alias). `ARC_AUDIT.md` Phase 1 marked DONE.

**Public API (recall):** `from smartflat.features.symbolic_barycenter.baselines import RTWE_NU, RTWE_LMBDA`.
Everything else is unchanged — no registry-key or public-signature change beyond the `max_iters` alias.

---

## 23. `baselines.py` split into distances/builders/registries/evaluation (Kickoff N, 2026-07-04)

**Branch `barycenter-baselines-split`** (off the Kickoff-M tip `d1164a3`; executes `ARC_AUDIT.md` **Phase 2**).
A pure **structural, behaviour-preserving** session — no new method, no changed default, no moved number.
The arc's 2131-line junk-drawer `baselines.py` (≥6 jobs, `ARC_AUDIT.md` Dimension 1 P1) is split into four
cohesive modules; `baselines.py` becomes a thin back-compat re-export shim so **not one `from …baselines
import X` changes**.

- **The partition (4 modules, byte-identical moves).** `distances.py` (433 L) — `RTWE_NU`/`RTWE_LMBDA`,
  embed/decode (`embed_symbolic_to_real`, `project_real_to_symbolic`, `_snap_and_reembed`, `_symbols_to_str`,
  `_classical_mds`), cost helpers (`ordinal_cost_matrix`, `_transition_matrix`), DTW/soft-DTW primitives, and
  every `dist_*`/`pmatch_*`; imports **nothing internal** (vendored `_rtwe`/`_eshape_dtw` stay lazy-imported
  inside functions). `builders.py` (926 L) — the 17 `barycenter_*` constructors (+ `_categorical_outer_loop`),
  importing embed/decode + primitives from `.distances`. `registries.py` (283 L) — the six `*_methods`
  `{build, distance}` registry builders, wiring `.builders` constructors to `.distances` scores.
  `evaluation.py` (508 L, the optional 4th file — kept separate to leave `registries.py` purely the method
  dicts) — CV helpers (`_make_clf`, `_nested_cv_auc`), evaluators (`evaluate_baselines`,
  `evaluate_incremental_ordering`, `baseline_significance_tests`), feature/label utilities, and the bootstrap
  CIs. **Dependency arrows are acyclic:** `distances → builders → {registries, evaluation}` (verified: no
  distances→builder back-edge; the one evaluation→builders edge is `evaluate_baselines` calling
  `barycenter_k_medoid`).

- **The shim contract.** `baselines.py` (97 L) = original docstring + `from .{distances,builders,registries,
  evaluation} import *` + **explicit** re-export of the 14 private-but-imported helpers (`import *` skips
  underscore names) + an `__all__` listing all **61** pre-split names (59 defs + 2 constants). Every symbol
  any repo file imported from `…baselines` (52 grepped, incl. `_make_clf`/`_nested_cv_auc`/`_pairwise_subsets`/
  `_transition_matrix`/`_snap_and_reembed`, imported by `order_evaluation.py`/`structure_metrics.py`/tests)
  still resolves. `barycenter_quality.py`'s `from …baselines import RTWE_NU, RTWE_LMBDA` and the M guard test
  keep working via the shim.

- **The gates (a pure move → import-surface + the frozen suite, not a numeric oracle).**
  `tests/test_baselines_public_surface.py` (written FIRST, green on the untouched tree) asserts every one of
  the 61 names is importable from the shim path. Every moved function was proven **byte-identical** against
  the git original via AST source-segment diff (59/59, full coverage, constants preserved). Frozen suite
  (11 M-suite files + `test_family_taxonomy` + the new surface test = 13 files) → **236 passed, 13 warnings**
  (was 234; +2 surface-test cases, warning count unchanged).

- **The one disclosed consequence — a stale white-box monkeypatch (not a production change).**
  `test_dba_dtw_cat_reference_is_integer_every_iteration` monkeypatched `baselines.embed_symbolic_to_real`
  to spy on re-embeds. Pre-split, `barycenter_dba_dtw`'s direct embed and its per-iter `_snap_and_reembed`
  both resolved to the single module global, so one patch caught both. Post-split they look up
  `embed_symbolic_to_real` in `builders` / `distances` respectively, so the shim patch no longer intercepts.
  The **production code is byte-identical**; the fix retargets the spy to both real lookup sites
  (`builders.embed_symbolic_to_real` + `distances.embed_symbolic_to_real`) — Python's "patch where it's
  looked up." It was the only monkeypatch of a baselines internal in the whole suite.

**Public API (recall):** all pre-split imports are unchanged — `from smartflat.features.symbolic_barycenter.baselines
import X` resolves for every X (public and private), and the concrete modules
(`…symbolic_barycenter.{distances,builders,registries,evaluation}`) are now also importable directly for new code.
