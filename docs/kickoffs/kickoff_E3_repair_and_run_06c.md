# Kickoff — Repair and execute `06c_g28_reproduction.ipynb` (E3)

*Reusable Claude Code kickoff. Paste as the first message of a fresh session. Self-contained.*

---

## Role & scope
You are working in the `smartflat` repo. **Read `PAPER_BRIDGE.md` first** (project rule for barycenter
work). Scope is one notebook, `notebooks/06c_g28_reproduction.ipynb` — the **G=28 data spine** that
`06`/`06b`/`06d` depend on. Surgical repair + a verified re-run. No library changes expected.

## Background — why 06c is broken (audit finding E3)
`docs/audits/notebook_content_audit_2026-07-15.md` §1 (E3): `06c` is the **last notebook in the
06-series still routing its barycenter through the uninstalled forked aeon**. `RESULTS_HANDOFF §26`
repaired `06` and `06b` onto `barycenter_mode_dba`; 06c was left behind. Concretely:

- **cell 1:** `from aeon.clustering.averaging import elastic_barycenter_average`
- **cell 10:** `idx_init, bary, best_cost, costs, idx_best = elastic_barycenter_average(..., distance='rtwe', precomputed_distances=…, integers=…)`

Stock aeon (0.11.1 in `smartflat_repro`, 1.2.0 elsewhere) returns a single ndarray, has **no**
`precomputed_distances`/`integers`/`nu`/`lmbda` params, and **rejects `distance='rtwe'`**. The
thesis-fork variant is not installed. So cell 10 cannot run; its committed output (`cost=12883.3…`) is
a **macOS-era artifact of a fork that no longer exists** (cell 1's committed output even prints a
`/Users/samperochon/` path).

Second, **independent** defect: 06c's hand-written **cell-17 verdict table** disagrees with the
`g28/auc_vs_L_faithful.csv` it owns on **33 of 34 numbers** (deltas up to +0.081). The tell: the
closed-form `wasserstein` drifts up to 0.025, which `RESULTS_HANDOFF §25` says is **impossible across
platforms** ("matches to 3 decimals") — so the table is **off-lineage**, not FP noise. The verdict
*direction* (frequency leads, alignment plateaus below) is unaffected.

## Goal (define done up front)
06c **executes clean end-to-end** in `smartflat_repro`, on the installed engine
(`barycenter_mode_dba` + `compute_alignment_path('rtwe')`), and its cell-17 verdict table is
**regenerated from the CSV it owns** (no hand-typed numbers). The G=28 build (pyramid path, 76.6%
coverage), the 28×28 temporal-occurrence `D_G`, and the frequency-leads verdict are preserved.

## The repair pattern (copy from what §26 already did to `06`)
Apply the same swap `06` used (see `06_barycenter_averaging.ipynb` cells for a worked reference):

- **cell 1:** remove `from aeon.clustering.averaging import elastic_barycenter_average`. The engine is
  already imported via `baselines as B` / `vocab`; use `B.barycenter_mode_dba` and
  `compute_alignment_path('rtwe')` (both already imported in 06c).
- **cell 10:** replace the `elastic_barycenter_average(distance='rtwe', precomputed_distances=D_G_cat,
  integers=…, nu=NU, lmbda=LMBDA, …)` call with
  `bary, costs = B.barycenter_mode_dba(Xg, D_G_cat, nu=NU, lmbda=LMBDA, max_iter=…, return_costs=True)`
  (mode/argmax categorical update — same math the paper describes; `return_costs=True` gives the
  per-iteration rTWE cost for the convergence figure). Alignment ribbons already go through
  `compute_alignment_path('rtwe', …)`.
- **cell 9 prose:** it describes the fork as "argmax/mode"; RH §12.1 notes the *thesis fork was
  mean-petitjean*. State plainly that the repaired engine is `barycenter_mode_dba`.
- **cell 15 gate:** the `# GATED OFF (~1 h)` comment predates the parallel evaluator; with
  `evaluate_baselines(n_jobs=…)` the full-length mode-DBA run is tractable (06b Tier B ran it in
  ~3.2 h). Either enable it with `n_jobs` or update the comment to say it is a serial-only gate.
- **cell 17 table:** stop hand-typing numbers — build the verdict table **programmatically from**
  `g28/auc_vs_L_faithful.csv`, so it can never drift again.

## Reproducibility contract
- Run in `smartflat_repro` (Python 3.11.15), `NUMBA_THREADING_LAYER=workqueue`, `MPLBACKEND=agg`.
- HP: `NU, LMBDA, OFFSET = 1e-4, 0.1, 0.3` (the shipped canonical point — 06c's current values; keep
  them, only fix the mislabelling comment "thesis G=28 hyperparameters" → "shipped canonical point,
  RTWE_NU/RTWE_LMBDA; thesis grid was nu=1e-5, PAPER_BRIDGE §6").
- Execute in-place via nbconvert **detached** so a disconnect can't kill it:
  `setsid bash -c 'nohup env MPLBACKEND=agg NUMBA_THREADING_LAYER=workqueue \
   <smartflat_repro>/bin/jupyter nbconvert --to notebook --execute --inplace \
   --ExecutePreprocessor.kernel_name=smartflat_repro --ExecutePreprocessor.timeout=7200 \
   notebooks/06c_g28_reproduction.ipynb >/tmp/06c_run.log 2>&1 &'`
- **Back up** `06c` + `g28/auc_vs_L_faithful.csv` before running; if the run changes any owned CSV
  value, stop and diff — the spine feeds 06/06b, so a shift there is a real event, not a cosmetic one.

## Verification (all must pass)
- 06c executes with **no CellExecutionError** end-to-end (the whole point of E3).
- The regenerated cell-17 table now **matches `auc_vs_L_faithful.csv`** (0 hand-typed deltas).
- **Cross-check invariant preserved:** 06c's L=5162 deterministic methods still match 06b's
  `baseline_comparison_g28.csv` to ≤1e-9 (wasserstein `0.809/0.812/0.746/0.571`, k_medoid
  `0.730/0.716/0.648/0.550`, majority_voting `0.772/0.654/0.613/0.540`).
- G=28 build unchanged: **27 categories + background via the pyramid path, 76.6% token coverage**
  (mapping `symb_labels` directly would give only 19 — must stay the pyramid build).
- `nbformat.validate` green; `git diff` shows only 06c source + its regenerated outputs/figures.
- Numbers cross-checked against the CSV, never from memory.

## Then update the record
- Mark E3 resolved in `docs/audits/notebook_content_audit_2026-07-15.md` and note in
  `RESULTS_HANDOFF_barycenters.md` that 06c is now on the installed engine (a §26 addendum / §27).

## Do NOT
- Reinstall or vendor a forked aeon — the repair is to use the shipped `barycenter_mode_dba`.
- Change the G=28 vocabulary, the pyramid build, or `D_G` (temporal-occurrence Wasserstein-1).
- Commit.
