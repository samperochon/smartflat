---
noteId: "8f9eaaa0351d11f196cbe54b89a7c6e4"
tags: []

---

# PAPER_BRIDGE.md — Barycenter Paper Code Needs

> This file bridges the standalone paper on barycenter averaging to the smartflat codebase. Read this before starting any barycenter-related work.

**Paper location:** `/Users/samperochon/These/manuscrit/ThesisManuscriptSamPerochon/paper-chapter-6-barycenters/`
**Paper status (2026-04-10):** 15-page full draft (Sections 1--8 + Abstract). Pre-submission after scientific review (Session 9). Score: 3.06/5 (major revision). All prose complete; **one TODO placeholder remains** at line ~397 of `main.tex` awaiting baseline comparison results from this repo.
**Paper-side documentation:** `COMPANION_CODE.md` (component/figure maps), `ROADMAP.md` (session plan), `barycenter_corpus/BASELINE_SHORTLIST.md` (full baseline specs).

---

## ⚠️ REPRODUCTION FINDINGS (server run, 2026-06-04) — READ BEFORE TOUCHING tab:baselines

Full record: **[RESULTS_HANDOFF_barycenters.md](RESULTS_HANDOFF_barycenters.md)**. NB06/NB06b were fixed and run end-to-end on the real data (`SymbolicSourceInferenceGoldConfig`, round 8). Key outcomes:

- **NB06 was previously dead** on non-HPC machines (`aeon.distances._rtwe` import → `HAS_AEON=False` skipped everything) and the `D_twe` regression dropped `D_G`. Both fixed; symbol `-1` noise remap + `D_G` restriction to occurring symbols added; grid search re-run (δ=0.3 confirmed, ν plateau, λ=0.5).
- **The proposed TW-TWE + DBA is NOT the best method on the K-space representation.** Faithful 10×3 AUCs: majority-voting wins Patient-vs-Control (0.74, **significantly** beats proposed, BH p=0.035); Wasserstein histogram wins RIL-vs-Control (0.84); edit-median wins TBI-vs-RIL (0.62). Proposed (mode-DBA + p_match) = 0.58/0.76/0.54.
- **Root cause:** the group signal is **symbol frequency** — a plain histogram reproduces the paper (0.84/0.57/0.80). The alignment-based barycenter does not improve on it. Also, **aeon's TWE-DBA mean-averages nominal symbol indices** (categorically wrong); a new `barycenter_mode_dba` (mode update) was added — it helps but still does not beat the frequency baselines.
- **Two ablations DO support the design** (now in the harness): mode-DBA ≫ mean-DBA (BH p=0.035) and Wasserstein cost ≫ ordinal cost.
- **The paper's 0.77/0.84 headline almost certainly uses the consolidated G≈28 semantic vocabulary** (A–J clinical scheme subcategories from the prototype visual annotation; *not* a column in the symbolization dataframe — the available label columns are all 76–178 fine prototypes). `tab:baselines` must be reproduced on that vocabulary before filling.
- **Open work** (kickoff prompts in RESULTS_HANDOFF §11): (1) reproduce on G≈28 + audit/improve TW-TWE+DBA vs majority voting; (2) consolidate a paper-improvement TODO (datasets/algorithms for a tier-1 venue) from the post-thesis docs.

### 🔄 UPDATE (2026-06-05) — G=28 reproduction DONE (RESULTS_HANDOFF §12; notebooks `06c`/`06d`; module `vocab.py`)

- **G=28 built faithfully** = 27 named cooking-action categories + background, from raw labels via the pyramid (`add_pyramid_labels` → `mapping_cluster_category` → segment-smooth), 76.6 % token coverage. Ground cost = a recomputed **28×28 temporal-occurrence Wasserstein** matrix (the `_cat_final` matrix was *not* on disk). The thesis barycenter's **forked aeon** (`elastic_barycenter_average` with `precomputed_distances`) is gone — reconstructed as `barycenter_mean_rtwe_dba`.
- **Finding confirmed + strengthened at G=28.** Histogram reproduces Patient-vs-Control (**0.79 ≈ 0.77**) and reaches **0.77** RIL-vs-Control; the **faithful TW-TWE+DBA barycenter is near-chance (0.46)**, mode 0.58/0.68. **No** method (bigram/ordering, Edit-Shape DTW, soft-mode DBA, re-selected HP) **BH-beats majority voting.** The paper's **0.84** matches the finer **G=77** histogram, not G=28.
- **Audit clarification:** passing `D_G` into the *averaging* does NOT rescue the index-mean (`mean_rtwe` 0.45 ≈ stock `aeon_mean` 0.54) — so the §3.2 "mean is broken" caveat is settled, independent of the alignment cost. p_match ≈ frequency overlap (r=0.63); mode-collapsed barycenters.
- **Implication for `tab:baselines`:** the honest table at G=28 shows frequency baselines competitive/superior — **do not** fill it with the proposed-method row. Reframe per §8 (the *design choices* are ablation-justified; the *group-discrimination claim* is not). One faint positive worth a powered follow-up: Edit-Shape DTW on **TBI-vs-RIL (0.59)**.
- **Next steps:** RESULTS_HANDOFF §12.6 (two tracks) and the new **Kickoff D** (settle ordering-vs-frequency) + **Kickoff C** (paper TODO consolidation) in §11.

Manuscript edits made (server copy): 4 inert `% REPRODUCTION FLAG` comments + 2 new figures; **no rendered content changed**.

---

## 1. What the paper needs from this repo

All items below correspond to the paper's "Session 5" backlog. Sub-sessions 5a, 5b, 5c, 5d, 5e, and 5f complete.

### Tier A -- Submission-blocking

| # | Item | Current state | Target |
|---|------|---------------|--------|
| A1 | **Unit tests for `_rtwe.py`** | ~~No tests exist~~ **DONE (5a)** | `tests/test_distances_rtwe.py` — 22 tests: symmetry, identity, triangle inequality, cost matrix shape, alignment-path correctness, pairwise, 1D/2D equivalence, edge cases, JIT smoke. Fixed bug: `rtwe_alignment_path_with_costs` used wrong index (`-2` → `-1`). |
| A2 | **Unit tests for `_eshape_dtw.py`** | ~~No tests exist~~ **DONE (5a)** | `tests/test_distances_eshape_dtw.py` — 14 tests: symmetry, identity, cost matrix shape, step_sequ effect, pairwise, alignment path. Fixed bug: `eshape_dtw_alignment_path` used raw sequence dims instead of cost matrix dims. |
| A3 | **Hyperparameter sweep port** | ~~Sketched in NB06~~ **DONE (5b)** | NB06 Section 1b: thesis grid `nus=[0,1e-5,1e-4] x lambdas=[0,1e-3,1e-2,1e-1]` + fine grid around final values. 500 pairs, alignment case boxplots. Saves `chapter_6_hyperparameters_searches.png`. |
| A4 | **NB08 figure reproduction** | ~~Stub~~ ~~PARTIAL (5b)~~ **DONE (5e)** — NB08 Ch.6 section now has 5 cells: data loading, distance matrix heatmap, DBA chronograms (bg/non_bg), medoid barycenters, classification report. All figures save to `{DATA_ROOT}/figures/thesis/`. | End-to-end run producing all `chapter_6_*.png` figures |
| A5 | **Baseline 1: DBA + standard DTW** (ablation) | ~~Not implemented~~ **DONE (5c)** | Pure numpy DBA with DTW alignment on D_G-row embeddings. In `baselines.py:barycenter_dba_dtw()`. |
| A6 | **Baseline 2: Soft-DTW barycenter** | ~~Not implemented~~ **DONE (5c)** | Gradient-descent Soft-DTW on D_G-row embeddings. In `baselines.py:barycenter_soft_dtw()`. |
| A7 | **Baseline 3: Edit-distance median string** | ~~Not implemented~~ **DONE (5c)** | Set median + iterative local search via python-Levenshtein. In `baselines.py:barycenter_edit_median()`. |

### Tier B -- Recommended

| # | Item | Difficulty |
|---|------|-----------|
| B1 | **Baseline 4: Wasserstein barycenter** (OT paradigm) | ~~MEDIUM~~ **DONE (5c/5d)** — `baselines.py:barycenter_wasserstein()` via POT |
| B2 | **Baseline 5: k-Medoid** (no averaging) | ~~LOW~~ **DONE (5c/5d)** — `baselines.py:barycenter_k_medoid()` |
| B3 | **Baseline 6: Majority voting** (lock-step) | ~~LOW~~ **DONE (5c/5d)** — `baselines.py:barycenter_majority_voting()` |

### Tier C -- Nice to have

| # | Item |
|---|------|
| C1 | ~~Baseline comparison notebook~~ **DONE (5e)** — `notebooks/06b_barycenter_baselines.ipynb`: 7-method evaluation (TW-TWE + 6 baselines), 10-split AUC comparison, chronogram visualization, saves `baseline_comparison.csv` |
| C2 | ~~Copy comparison figures~~ **DONE (5e)** — figures saved to `{DATA_ROOT}/outputs/symbolic_barycenter/` and `{DATA_ROOT}/figures/thesis/` (not copied to paper repo per constraint) |

---

## 2. Baseline specifications

### Baseline 1: DBA + Standard DTW (Ablation) -- `must_implement`, LOW

- **Citation:** Petitjean et al. (2011). *Pattern Recognition* 44(3), 678--693.
- **What it isolates:** Contribution of the Wasserstein-based pointwise distance (TW-TWE vs. standard DTW).
- **Implementation:** Swap distance parameter in `aeon.clustering.averaging.elastic_barycenter_average` from TW-TWE to DTW. Requires mapping symbols to real-valued temporal embeddings (e.g., temporal occurrence centroids) for DTW's L2 cost.
- **Library:** `aeon`
- **Expected vs. TW-TWE+DBA:** Likely loses -- DTW lacks edit operations, may produce degenerate alignments for symbolic sequences.

### Baseline 2: Soft-DTW Barycenter -- `must_implement`, LOW

- **Citation:** Cuturi & Blondel (2017). *ICML*, 894--903.
- **What it isolates:** Hard-alignment (DBA) vs. soft-alignment (differentiable) paradigm.
- **Implementation:** `tslearn.barycenters.softdtw_barycenter`. Same real-valued embedding as Baseline 1.
- **Library:** `tslearn`
- **Expected vs. TW-TWE+DBA:** Likely loses on symbolic data -- smoothing parameter gamma blurs categorical boundaries.

### Baseline 3: Edit-Distance Median String (Pivot Heuristic) -- `must_implement`, MEDIUM

- **Citation:** Mirabal et al. (2019). *SCCC*, 1--5. Problem definition: Kohonen (1985).
- **What it isolates:** Whether temporal structure helps symbolic averaging (pure string vs. temporal-elastic).
- **Implementation:** Custom Python. (1) Levenshtein distance for G=28 alphabet via `python-Levenshtein`, (2) set median initialization, (3) iterative local search (insert/delete/substitute at each position).
- **Library:** `python-Levenshtein` (+ custom code)
- **Expected vs. TW-TWE+DBA:** Likely loses on classification -- ignores temporal structure. Interesting for pure symbolic consensus.

### Baseline 4: Wasserstein Barycenter -- `recommended`, MEDIUM

- **Citation:** Peyre & Cuturi (2019). *Foundations and Trends in ML* 11(5--6).
- **What it isolates:** Distributional averaging (histograms) vs. sequence-level averaging.
- **Implementation:** `POT` library `ot.barycenter`. Represent each sequence as 28-dim histogram (symbol frequencies weighted by duration).
- **Library:** `POT`

### Baseline 5: k-Medoid -- `recommended`, LOW

- **What it isolates:** Whether constructing a synthetic average adds value over selecting the best existing sequence.
- **Implementation:** `numpy.argmin` on existing TW-TWE distance matrix row sums per group.
- **Library:** numpy only

### Baseline 6: Majority Voting (Lock-Step) -- `recommended`, LOW

- **What it isolates:** Whether elastic alignment adds value over per-timestamp voting.
- **Implementation:** Already in thesis (`chapter_6.tex` line 546). Port to refactored notebook.
- **Known result:** Produces "degenerated barycenters" (thesis finding). Confirms elastic alignment is necessary.

---

## 3. Scientific review findings that affect code work

The Session 9 scientific review (score 3.06/5, major revision) identified findings that require code-side action:

### CRITICAL (submission-blocking)

| Finding | Code implication |
|---------|-----------------|
| **Tau_k classification leakage risk** | ~~Document split protocol~~ **DONE (5f)** — Markdown cell + `assert len(set(train_idx) & set(test_idx)) == 0` added to NB06 before the train/test split cell. `evaluate_baselines()` in `baselines.py` also enforces this independently. |
| **Demographics table absent** | ~~Generate table~~ **DONE (5f)** — NB07 cell `zmpq4kxwyjb` generates demographics (age, sex, MoCA, ISDC, duration). Education and time-since-injury documented as unavailable in SDS2 protocol (note cell added after demographics). |
| **Sample-size justification absent** | Not directly a code task, but any power analysis or bootstrap CI would run here. |

### HIGH

| Finding | Code implication |
|---------|-----------------|
| **DBA convergence diagnostics** | ~~Add convergence curve~~ **DONE (5b)** — NB06 tracks cost vs. iteration for all 3 groups (20 iterations), saves `chapter_6_dba_convergence.png` + `convergence.csv`. |
| **Supplementary incomplete** | ~~Baseline figures~~ **DONE (5e)** — NB06b produces `baseline_comparison_auc.png`, `baseline_chronograms.png`, `baseline_wasserstein_histograms.png`. NB08 produces `chapter_6_barycenters_median_twe.png`. |
| **Software/code availability statement** | Draft in `COMPANION_CODE.md` section 5. Will reference this repo by name. |

### MEDIUM

| Finding | Code implication |
|---------|-----------------|
| **Sequence length distribution** | ~~Add histogram~~ **ALREADY EXISTS** — NB07 cell `pn7ocz1tqt` has ECDF per pathology + LME model with ICC. |
| **BH correction alpha** | ~~Verify~~ **DONE (5b)** — `alpha=0.05` now explicit in `multipletests()` call in NB07 cell `d2mefj7jeaj`. |

---

## 4. Figure-to-notebook map

Figures the paper needs, and which notebook must produce them:

| Figure filename | Paper section | Notebook | Status |
|----------------|---------------|----------|--------|
| `chapter_6_hyperparameters_searches.png` | Section 3.3 | NB06 Section 1b (sweep ported in 5b) | **DONE (5b)** |
| `chapter_6_distance_matrix_pairwise.png` | Section 5.1 | NB08 cell `ch6-distance-matrix` | REPRODUCIBLE |
| `chapter_6_non_bg.png` | Section 5.2 | NB08 cell `ch6-barycenter-chronograms` | REPRODUCIBLE |
| `per_label_pathologie_p_values.png` | Section 5.3 | NB07 | REPRODUCIBLE |
| `chapter_6_classification_report.png` | Section 5.4 | NB08 cell `ch6-classification-report` | REPRODUCIBLE |
| `chapter_6_bg.png` | Supplementary Section 1 | NB08 cell `ch6-barycenter-chronograms` | REPRODUCIBLE |
| `chapter_6_barycenters_median_twe.png` | Supplementary Section 2 | NB08 cell `ch6-medoid-barycenters` | **DONE (5e)** |
| `per_label_group_p_values.png` | Supplementary Section 3 | NB07 | REPRODUCIBLE |
| *Baseline comparison figure(s)* | Supplementary (new) | `06b_barycenter_baselines.ipynb` | **DONE (5e)** |

---

## 5. Coordination protocol

1. **When code work completes**, update `paper-chapter-6-barycenters/COMPANION_CODE.md`:
   - Check off completed TODO items in Section 4
   - Add version history entry
   - Update component/figure status columns

2. **Comparison figures** go to `paper-chapter-6-barycenters/figures/` with provenance noted in `COMPANION_CODE.md`.

3. **The paper's Session 5 TODO placeholder** (line ~397 of `main.tex`) will be filled with baseline comparison results once the code work is done.

4. **Do not edit** `main.tex` or any LaTeX files from this repo -- paper edits happen in the paper project.

---

## 6. Key technical context

- **Cohort:** 122 participants (Control n=26, TBI n=59, RIL n=37)
- **Symbol vocabulary:** G=28 categories (cooking actions derived via recursive prototyping)
- **Sequence length:** Interpolated to median L=5,182 symbols (~27.4 min)
- **Distance:** TW-TWE (Temporal-Wasserstein Time Warp Edit) -- rTWE with outer Edit-Shape DTW
- **Barycenter:** DBA with mode-based update (mode replaces mean for categorical data)
- **Hyperparameters:** nu=1e-5, lambda=0.1, delta=0.3 (from grid search)
- **Evaluation:** 10 random 50/50 train-test splits, 3 initializations per split, AUC-ROC with p_match feature
- **Key result:** AUC 0.77 (Patient vs Control), 0.84 (RIL vs Control), 0.59 (TBI vs RIL)

---

## 7. Suggested sub-session breakdown

Session 5 is expected to span multiple sub-sessions:

1. **5a: Unit tests** -- A1, A2 (foundation for confident baseline work)
2. **5b: Hyperparameter sweep + NB08** -- A3, A4 (figure reproducibility)
3. **5c: Must-implement baselines** -- A5, A6, A7 (core comparison)
4. **5d: Recommended baselines** -- B1, B2, B3 (strengthens paper)
5. **5e: Comparison notebook + figures** -- C1, C2 (final deliverable back to paper)
6. **5f: Review-driven additions** -- convergence diagnostics, demographics table, sequence length histogram

---

*Created: 2026-04-10 (paper Session 9.5 -- bridge propagation). To be updated as Session 5 sub-sessions complete.*
