# Barycenter Baseline Comparison — Results Handoff

**Run on:** the Smartflat data server (`SMARTFLAT_DATA_ROOT=/home/perochon/data-gold-final`), round 8, `SymbolicSourceInferenceGoldConfig`.
**Scope:** fix and run NB06/NB06b end-to-end; re-select TW-TWE hyperparameters; produce the baseline comparison + ablation + significance + stability for the paper.
**Headline:** the pipeline now runs end-to-end, but a faithful reproduction surfaces a substantive finding that bears on the paper's §5.5 claim — documented below. **No results were tuned to make the proposed method win.**

> ⚠️ **Flag for the authors (do not silently integrate):** on this data, with the K-space symbolic representation and a faithful pipeline, the proposed **TW-TWE + DBA** method is **competitive only on RIL-vs-Control** and is **matched or beaten by simple frequency baselines (Wasserstein histogram, majority voting)** on the pooled Patient-vs-Control comparison. The group-discriminative signal is dominated by **which prototypes occur (symbol frequency)**, not their temporal alignment. See §3.

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

### Kickoff C — "Consolidate the paper-improvement TODO for a tier-1 venue" (separate fresh session)
> Work in `/home/perochon/paper-chapter-6-barycenters`. Synthesise the prior audit/review work into a single
> living TODO document for raising the paper to tier-1 acceptability. Sources: `ROADMAP.md`,
> `COMPANION_CODE.md`, `barycenter_corpus/BASELINE_SHORTLIST.md`, `grounding-memory-log.md`, the
> scientific-reviewer output (3.06/5, major revision), and any post-thesis audit notes. Produce/maintain a
> `PAPER_TODO.md` covering: (1) **additional datasets** to add (the §7 future-work mentions Breakfast,
> BPI-Challenge-2012; assess generalisation beyond n=122), (2) **additional baselines/algorithms**
> (ShapeDBA, block-level edit distances, GW averaging), (3) **the reproduction findings above** (frequency
> baselines competitive → either strengthen the method or reframe the contribution), (4) reviewer concerns,
> (5) prioritisation for submission. Cross-link `smartflat/RESULTS_HANDOFF_barycenters.md`.
