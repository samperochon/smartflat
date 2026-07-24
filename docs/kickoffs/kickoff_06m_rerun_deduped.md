# Kickoff — Re-run `06m_generalization_breakfast.ipynb` on the deduped set (E4)

*The code fix is already applied; this kickoff only executes and verifies it. Paste into a fresh
session **when you are done hand-editing 06m** (see the concurrency note).*

---

## Status: the fix is done, the re-run is not
The leakage fix from audit finding E4 is **already wired into `06m` source**:
- `cell 1` imports `dedup_by_execution`.
- `cell 3` calls `meta, X, labels, _dedup_idx = dedup_by_execution(meta, X, labels, NAME)` right after
  `load_action_seg`, and the sanity display now reports **independent executions** + the raw count.
- The prose (cells 0, 6, 7, 14) already carries the corrected framing (frequency-saturation diagnosis;
  leakage biases ΔAUC *upward* so an all-null verdict is conservative; the narrowed paper claim).

**What is left is the re-execution** — the committed `order_null.csv` / `quality.csv` are still the
**pre-dedup run** (per-file, 1712 videos), so `06m`'s source and its outputs are currently
inconsistent by design until you run this.

## Why this was not auto-run in the audit session (read before launching)
`06m` was under **active parallel editing** during the audit — it grew from 13 → 15 cells (a
"frequency-headroom screen", cells 8–9) and `06n_frequency_controlled_order.ipynb` appeared. A
detached `--inplace` re-execution takes **hours** (the order-null cell alone exceeded the 1800 s cap;
the notebook budgets it at ~6.8 h) and would **overwrite any hand-edits made while it runs**. So:
**only launch this when you have stopped editing 06m**, and re-read cell 3 first to confirm the dedup
line is still present (a manual save could have reverted it).

## Derisk already done (you don't need to re-check)
`dedup_by_execution` on real Breakfast GT: **1712 videos → 503 executions**, all 10 activity classes
retained, every top-6 class has **49–52** executions — all comfortably above the order-null's 5-fold
minimum and the quality half's ≤50/class subsample. So the re-run will not drop a class or under-power
a fold.

## Run it (detached, disconnect-proof)
```bash
cp notebooks/06m_generalization_breakfast.ipynb /tmp/06m_backup.ipynb
cp /home/perochon/data-gold-final/outputs/symbolic_barycenter/generalization/breakfast/*.csv /tmp/
setsid bash -c 'nohup env MPLBACKEND=agg NUMBA_THREADING_LAYER=workqueue \
  <smartflat_repro>/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=smartflat_repro --ExecutePreprocessor.timeout=30000 \
  notebooks/06m_generalization_breakfast.ipynb >/tmp/06m_run.log 2>&1 &'
```
(Use the `smartflat_repro` env — the kernel that produced 06m's committed outputs, Python 3.11.15. A
30000 s per-cell timeout clears the ~6.8 h order-null cell.)

## Verification (all must pass)
- **Executes clean** end-to-end, no `CellExecutionError`.
- **Cell 3 now reports executions:** `N independent executions (deduped) = 503`, `N raw camera-view
  videos = 1712`.
- **The verdict is expected to stay `order_helps` 0/60** — but for the *right* reason now: with the
  leak removed, `auc_intact` is **no longer pinned at 1.0** for every cell (that was the duplicate-view
  memorisation channel). Record the new `auc_intact` distribution. If any cell flips to `order_helps =
  True`, **stop and investigate** — that would be a real (leak-free) order signal, a finding, not a
  routine refresh.
- **Direction-of-bias argument holds:** leakage could only manufacture false *positives*, so a
  still-null result on the deduped set is strictly stronger evidence than the pre-dedup 0/60. Update
  the cell-14 Result narrative to report the deduped `auc_intact` range and drop any residual "AUC=1.0
  on all cells" phrasing that no longer describes the output.
- **Cross-check with the headroom screen (cells 8–9):** those measure histogram saturation
  independently; confirm the deduped run tells the same saturation story (Breakfast's top-6 vocab is
  near-disjoint), so 06m and the headroom cells + `06n` stay mutually consistent.
- `nbformat.validate` green; `git diff` shows 06m source (already applied) + regenerated
  `order_null.csv`/`quality.csv`/`chronograms.png`.

## Then update the record
- Fold the deduped 06m result into the (still-missing) **`RESULTS_HANDOFF §27`** for the Phase-2
  action-seg family (E5), with the per-execution N and the leak-free `auc_intact` range.
- Commit `generalization/headroom.py`, `tests/test_headroom.py`, and the `dedup_by_execution` addition
  to `generalization/action_segmentation.py` (all currently **untracked/uncommitted**) so 06m's and
  06n's imports resolve on a clean checkout.
