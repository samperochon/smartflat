# Kickoff — Recreate `tab:baselines` at the notebook-selected HP optimum (E1)

*Reusable Claude Code kickoff. Paste as the first message of a fresh session. Self-contained; every
number to match is pinned below so success is verifiable, not judged.*

---

## Role & scope
You are working in the `smartflat` repo on the barycenter paper's baseline table. **Read
`PAPER_BRIDGE.md` before touching any barycenter code** (project rule). This task is **surgical**: one
notebook (`notebooks/06b_barycenter_baselines.ipynb`), one added artifact, no library changes unless a
test forces one. Do not refactor adjacent code.

## Background — the problem you are fixing (audit finding E1)
`docs/audits/notebook_content_audit_2026-07-15.md` §1 (E1) found that the G=28 baseline table is **not
computed at the operating point the HP search selects**:

- `06_barycenter_averaging.ipynb` cell 8 runs the G=28 grid and **re-selects**
  **`nu=1e-3, lmbda=0.1, offset=0.1`** (committed output: `selected: nu=0.001 lmbda=0.1 offset=0.1
  (inner-CV AUC=0.680)`). Its AUC-vs-L sweep (`g28/auc_vs_L_g28.csv`) runs at that point.
- `06b` (the `tab:baselines` source) and `06c` (the data spine) run at the **shipped canonical point
  `nu=1e-4, lmbda=0.1, offset=0.3`** (`RTWE_NU`/`RTWE_LMBDA` in
  `smartflat/features/symbolic_barycenter/distances.py`).
- Consequence: `RESULTS_HANDOFF_barycenters.md` §26 quotes two different `wasserstein` values for
  "G=28 at L≈5162" — 0.79/0.77 (06's sweep) vs 0.81/0.81 (06b's table) — without saying they are
  different operating points. Diagnostic proof it is HP-driven: `majority_voting` (scored by
  `dist_hamming`, touches no `D_G`/nu/lmbda) is **identical to the digit across all three caches**;
  only the HP-dependent methods diverge.

## Goal (define done up front)
Produce the baseline table **at the notebook-selected optimum `nu=1e-3, lmbda=0.1, offset=0.1`**,
**alongside** the existing canonical-point table (do not overwrite it), so the paper can either report
the selected-optimum result or carry it as a sensitivity row — and so §26's two-values ambiguity is
resolved explicitly. Everything reproducible and cross-checked.

## Reproducibility contract (non-negotiable)
1. **Kernel/env:** run in the `smartflat_repro` conda env (Python 3.11.15), the same env that produced
   the committed outputs (`requirements-lock.txt`: tslearn 0.6.4, POT 0.9.6.post1, numba 0.60.0,
   scikit-learn 1.5.2, scipy 1.12.0, numpy 1.26.4). Set `NUMBA_THREADING_LAYER=workqueue`,
   `MPLBACKEND=agg`.
2. **Determinism:** use `evaluate_baselines(..., n_jobs=N)` — it derives each unit's seed from
   `(split_idx, init_idx)`, so results are **bit-identical to serial** (§26 verified `n_jobs=1` vs `4`
   → `assert_frame_equal` exact). Keep the same protocol as the canonical run: **10 stratified 50/50
   splits × 3 inits**, `random_state` unchanged, full length **L=5162** for the mode-DBA family +
   edit-median + Wasserstein + k-medoid + majority-voting; the pure-Python `dba_dtw`/`soft_dtw` stay
   budget-gated (`dba_dtw` L=256, `soft_dtw` L=128) exactly as now.
3. **No overwrite:** write the selected-optimum results to a **new** file
   `g28/baseline_comparison_g28_selHP.csv` (and `..._significance_g28_selHP.csv`). Leave
   `baseline_comparison_g28.csv` (the canonical-point table) untouched.
4. **Single source of the HP:** define `NU, LMBDA, OFFSET = 1e-3, 0.1, 0.1` in one cell with a comment
   citing 06 cell 8's selection; do not scatter literals.

## Implementation steps
1. Read `06b` cells 1–8 and `06`'s cell 8 (the HP-selection cell) so you reuse the exact grid result,
   not a guessed value. Confirm 06 selects `1e-3/0.1/0.1`.
2. In `06b`, add a **parameterised second pass**: factor the Tier-B roster build so it runs once at the
   canonical point (existing) and once at `1e-3/0.1/0.1`, writing the `_selHP` CSVs. Reuse
   `evaluate_baselines` and `baseline_significance_tests` from
   `smartflat/features/symbolic_barycenter/evaluation.py` — do **not** re-implement.
3. Add a short markdown cell that tabulates **canonical vs selected-optimum** side by side and states
   plainly which the paper will report. Note the expected direction: `offset=0.1` sharpens the ground
   cost, `nu=1e-3` stiffens rTWE — the frequency `wasserstein` row is HP-independent and **must not
   move** between the two tables (use it as a built-in control).

## Verification (must all pass before you call it done)
- **HP-independence control:** `wasserstein` and `majority_voting` rows are **identical** (to ≥3 dp)
  between `baseline_comparison_g28.csv` and `baseline_comparison_g28_selHP.csv`. If they differ, the
  run is wrong — stop and diagnose (you changed something that shouldn't depend on HP).
- **Cross-check invariant:** the deterministic cheap methods at L=5162 in the new table match 06c's
  `g28/auc_vs_L_faithful.csv` exactly where the HP doesn't enter (wasserstein
  `0.809/0.812/0.746/0.571`, majority_voting `0.772/0.654/0.613/0.540`, k_medoid at the *canonical*
  point `0.730/0.716/0.648/0.550`).
- **Bit-identical determinism:** run the new pass at `n_jobs=1` and `n_jobs>1`; `assert_frame_equal`
  must pass.
- **Re-run stability:** execute the notebook twice; the `_selHP` CSVs must be byte-identical.
- **No collateral change:** `git diff` shows only `06b` source + the two new CSVs (+ figures if you add
  a bars plot). Committed outputs of every other cell unchanged.
- Notebook parses (`nbformat.validate`) and executes clean end-to-end in `smartflat_repro`.

## Then update the record
- Append the selected-optimum table to `RESULTS_HANDOFF_barycenters.md` (a short §26 addendum or §27)
  stating the two operating points explicitly and which the paper reports.
- Note in `docs/audits/notebook_content_audit_2026-07-15.md` that E1 is resolved (how, and where).
- If the paper is to report the selected-optimum numbers, flag `PAPER_TODO` §1.3/§6.1 (`tab:baselines`
  fill) accordingly — but do **not** edit `main.tex` unless asked.

## Do NOT
- Overwrite the canonical-point CSVs or figures.
- Change `RTWE_NU`/`RTWE_LMBDA` in `distances.py` (those are the shipped constants; this task is about
  the *table*, not the library default).
- Commit (leave the working tree for the user to review).
