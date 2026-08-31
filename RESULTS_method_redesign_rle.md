# §28 — Method redesign: segment-scale, duration-explicit (RLE) TW-TWE + DBA

*Session 2026-08-17. Standalone results-of-record (RESULTS_HANDOFF_barycenters.md is
uncommitted WIP; this follows the §27 / RESULTS_frequency_controlled_order.md precedent).
Plan: `~/.claude/plans/greedy-hatching-zebra.md`. Artifacts:
`$DATA_ROOT/outputs/symbolic_barycenter/g28/redesign/` (git-ignored CSVs; this file is the
committed readable record).*

## 28.1 Motivation — the frequency-dominance result is partly built into the method

Code inspection established that four design choices bias the proposed frame-level
TW-TWE + mode-DBA toward reproducing the histogram it is evaluated against:

1. **Representation scale.** Alignment runs on frame-level sequences (L≈5170) with
   immediate-repeat rate 0.95–0.97 and mean run length 20–34 — elastic alignment of
   ~25×-redundant runs mostly stretches identical-symbol runs, which is approximately
   order-insensitive. The order signal lives at the segment scale (~180 runs).
2. **Flat ground cost.** The thesis recipe adds +0.3 to the *raw* W1 matrix (max ≈ 0.48)
   before max-normalizing — a 63%-of-scale floor compressing off-diagonal costs to
   [0.39, 1]. Which symbol is substituted barely matters.
3. **"When" conflated with "what".** The temporal-occurrence ground cost gives
   temporally co-occurring but semantically unrelated actions near-zero cost; the
   embedding geometry that could separate them is discarded at symbolization.
4. **Frequency-favoring update.** The `Counter` mode update ignores `D_G` entirely.

Two method-integrity issues were also found: the cohort-level `D_G` is estimated on
**all** administrations (including future test folds — a leak), and
`barycenter_mode_dba` was **deterministic** (medoid init, `random_state` unused) while
the protocol claims 3 random inits, and returned the *last* iterate rather than the
best. A suspected fifth issue — the rTWE `_pad_arrs` padding with symbol 0 (= background)
— was checked and **downgraded**: with the default zero-diagonal ground cost the only
reachable pad lookup is `D[0,0] = 0`, so there is no boundary artifact (it is real only
for the `zero_diagonal=False` verbatim-thesis variant).

## 28.2 What shipped (library, all unit-tested; 320/320 tests pass)

- **`engine/distances/_rtwe_duration.py`** — duration-aware rTWE for RLE sequences:
  pointwise cost `D_G[a,b] + γ·|log d_a − log d_b|` (a metric when `D_G` is; TWE metric
  property preserved). Neutral sentinel padding (correct for *any* ground cost). No
  resampling anywhere: ragged sequences are consumed directly. `γ=0` reduces **exactly**
  to the vendored rTWE on the RLE symbols (pinned by test).
- **`features/symbolic_barycenter/rle.py`** — `rle_encode/decode`, `rle_object_array`
  (harness-compatible packing), `dist_rle_twe`, `pmatch_rle` (duration-weighted, optional
  soft credit), `barycenter_rle_dba` (D_G-Fréchet or mode update + geometric-mean
  duration per position; best-iterate return; adjacent-run merging → adaptive length),
  `rle_methods` registry.
- **`builders.barycenter_mode_dba`** gained `update={'mode','dg_frechet'}`,
  `init={'medoid','random'}` (seed finally used), `keep={'best','last'}` (default
  `'best'`; coincides with `'last'` under the empirically monotone trace). Historical
  behaviour = `update='mode', keep='last'` (regression-tested).
- **`vocab.temporal_ground_cost(X, G)`** — leakage-free temporal cost as a pure function
  of a sequence set (train-fold-only / control-only estimation now possible). Exact
  empirical **W1** (no KDE, no bandwidth, no OT solver) — note the cached thesis matrix
  is actually `ot.emd2_1d(metric='sqeuclidean')`, not the W1 the paper describes;
  the two correlate at **r = 0.967** on G=28. `vocab.semantic_ground_cost` (category
  cosine from prototype centroids) + `vocab.blend_ground_costs` (the "what vs when" α
  knob) are implemented and tested; the K-space centroid *data wiring* is still to do
  (round-8 `clusterdf.pkl` holds stats only — needs `compute_reduced_centroids`).
- **`vocab.compute_distance_matrix(method='offdiag_pre')`** — scale-calibrated offset
  (normalize first, offset off-diagonals only): floor becomes a known fraction of the
  symbol-geometry scale (G=28 contrast [0.39,1] → [0.23,1]).
- **`evaluation.evaluate_baselines(methods_factory=...)`** — per-fold method registries,
  enabling leakage-free ground costs inside the CV loop.
- **`distances.pmatch_to_barycenter(denominator={'diagonal','all'})`** +
  `pmatch_soft_to_barycenter` (D_G-weighted soft agreement) — the two p_match
  definitions in circulation are now explicit options.
- Tests: `tests/test_rle_barycenter.py` (22 tests: RLE round-trip, γ=0 exact
  equivalence, triangle inequality of the (symbol, duration) cost, update/init/keep
  contracts, W1-vs-scipy exactness, factory equivalence).

**Compute:** RLE median L = 178 (min 118, max 301) vs 5170 → each pairwise alignment
~800× cheaper. The full 10-split, 8-method suite runs in **82 s** on a laptop; the
frame-level §26 equivalent took ~3.2 h on a server. Everything §25 gated (soft-DTW, MSA,
quality axes at full fidelity) is now un-gated at this scale.

## 28.3 Experiment 1 — RLE methods on SDS2 G=28 (10 splits, seed 42, ν=1e-3, λ=0.1)

Reference points (frame-level, §26, same cohort): `tw_twe_pmatch` 0.69 / 0.76 / 0.53,
`tw_twe` 0.73 / 0.76 / 0.57, `wasserstein` histogram 0.81 / 0.81 / 0.57
(Control-v-Patient / HEALTHY-v-RIL / RIL-v-TBI). The in-run `wasserstein` control
(0.80 / 0.79 / 0.58) reproduces §26 within noise, confirming the shared spine.

| method (RLE, γ) | C-v-P | H-v-RIL | H-v-TBI | RIL-v-TBI |
|---|---|---|---|---|
| rle_dba_pmatch γ=0 | 0.57 | 0.64 | 0.51 | 0.52 |
| rle_dba_pmatch γ=0.5 | 0.67 | 0.70 | 0.61 | 0.54 |
| rle_dba_twe γ=0 | 0.66 | 0.72 | 0.66 | 0.55 |
| **rle_dba_twe γ=0.5** | **0.75** | **0.84** | 0.64 | 0.61 |
| rle_dba_twe γ=1.0 | 0.61 | 0.84 | 0.57 | 0.61 |
| **rle_dba_twe γ=2.0** | 0.62 | 0.80 | 0.50 | **0.64** |
| rle_medoid γ=0.2 | 0.64 | 0.70 | 0.65 | 0.51 |
| wasserstein (control) | 0.80 | 0.79 | 0.74 | 0.58 |

Readings:
- **Durations carry real signal**: every feature improves monotonically in γ from γ=0
  up to a comparison-specific optimum; γ=0 (pure segment-scale symbols) is *worse* than
  frame level, because frame-level p_match was implicitly duration-weighted. The
  redesign moves that signal from an artifact of the representation into an explicit,
  tunable term of the metric.
- **The duration-aware distance (`rle_dba_twe`) is the first alignment method to reach
  the histogram**: 0.84 on HEALTHY-v-RIL (the paper's original headline number, now from
  a principled method at 1/800 the compute). The apparent 0.61–0.64 on RIL-v-TBI did
  not survive the fresh-seed confirmation — see §28.4.
- p_match remains the weaker feature at this scale; the raw distance (which sees
  durations and edit structure, not just symbol agreement) is the right native feature.

## 28.4 Honest selection — leave-one-split-out γ, paired vs the histogram

Sweeping γ on test AUC is the same circularity §5's hyperparameter selection was
criticized for, so the confirmatory analysis selects γ per held-out split from the
other 9 splits only (LOSO; `exp1c_loso_gamma_vs_wasserstein.csv`):

| comparison | LOSO rle_dba_twe | wasserstein | Δ | Wilcoxon p (paired, 10 splits) |
|---|---|---|---|---|
| CONTROL_vs_PATIENT | 0.748 | 0.804 | −0.055 | 0.027 (histogram better) |
| HEALTHY_vs_RIL | 0.809 | 0.792 | +0.017 | 0.625 (parity) |
| HEALTHY_vs_TBI | 0.692 | 0.742 | −0.051 | 0.084 |
| **RIL_vs_TBI** | **0.642** | 0.579 | **+0.063** | **0.027** (γ=2.0 chosen in every fold) |

BH over the 4 comparisons adjusts both p=0.027 values to ≈0.054 — **marginal**. Because
the γ grid itself had been explored on these seed-42 splits, a pre-registered-style
confirmation was run on **20 fresh splits (seed 1234)**, LOSO-γ + paired Wilcoxon + BH
(`exp1d_confirmation_20splits_seed1234.csv`, `exp1d_loso_confirmation.csv`):

| comparison | LOSO rle_dba_twe | wasserstein | Δ | p_BH |
|---|---|---|---|---|
| CONTROL_vs_PATIENT | 0.710 | 0.783 | −0.073 | 0.009 (histogram better) |
| **HEALTHY_vs_RIL** | **0.838** | 0.826 | +0.012 | 0.701 (**parity, replicates**) |
| HEALTHY_vs_TBI | 0.673 | 0.759 | −0.087 | <0.001 (histogram better) |
| RIL_vs_TBI | 0.552 | 0.616 | −0.064 | <0.001 (histogram better; **seed-42 win REFUTED**) |

The defensible conclusions after confirmation:

1. **The seed-42 RIL-v-TBI "alignment beats frequency" result did NOT replicate** — on
   fresh splits the histogram wins that contrast significantly (its seed-42 value of
   0.579 was itself a low draw; it moved to 0.616). The LOSO-chosen γ was also unstable
   there ({0.5, 1.0, 2.0} across folds). Killed by our own confirmation run; recorded,
   not softened. Split-level variance at n=96 is large enough that a single 10-split
   comparison can flip sign — a warning that applies to the §26 table too.
2. **What replicates: HEALTHY-v-RIL parity at a higher level.** 0.838 vs 0.826 on fresh
   splits (0.809 vs 0.792 on seed 42) — the duration-aware segment-scale distance is the
   first alignment-based method to *match* the frequency histogram on this contrast
   (frame-level `tw_twe` trailed at 0.76), with γ=0.5 chosen in every fold.
3. **Frequency dominance otherwise stands**, now confirmed against a fairer opponent:
   the histogram significantly leads the pooled and both TBI-involving comparisons.

## 28.5 Experiment 2 — ground-cost grid (honest nulls)

`rle_dba_pmatch` γ=0.2, W1-estimator ground cost, {thesis recipe vs `offdiag_pre`} ×
{full-cohort vs train-fold-only estimation} (`exp2_ground_cost_grid.csv`): all cells
within ±0.03 AUC of each other (e.g. H-v-RIL 0.61–0.65). Two conclusions:

- **The full-cohort estimation leak is real but small** (~1 pt) — fixing it (now
  possible via `methods_factory`) does not change any conclusion, but the leakage-free
  path should be the default in any paper protocol.
- **The dynamic-range fix does not translate into AUC** at this operating point —
  the flat-cost critique (28.1 §2) is structurally correct but empirically inert here.
  Reported as a null, not softened.

## 28.6 Experiment 3 — D_G-Fréchet vs Counter-mode update (null)

`update='dg_frechet'` vs `'mode'` at γ=0.2, same builder/feature otherwise: 0.62/0.66/0.54
vs 0.63/0.67/0.55 — no significant difference (all BH n.s.). With G=28 and strong
per-position modes the two rules mostly agree; the Fréchet update is kept as the
principled default in `rle.py` (it is objective-consistent and costs nothing) but it is
**not** an empirical winner. Honest null.

## 28.7 Caveats

- ν=1e-3, λ=0.1 were carried from the §26 G=28 optimum, not re-tuned at the RLE scale
  (index semantics changed ~29×); a joint nested (γ, ν, λ) selection is the clean next
  step and could move numbers in either direction.
- Single dataset; the generalization corpus (06l–06n) has not yet been re-run with the
  RLE methods.
- Age/sex confounds (RIL older, sex imbalance) remain unaddressed — durations plausibly
  correlate with age; a covariate-adjusted analysis is still owed before clinical claims.

## 28.8 Next steps

1. Fold the confirmed results (HEALTHY-v-RIL parity, everything-else-histogram) into
   the paper's honest `tab:baselines` as the proposed method's segment-scale variant,
   with a nested (γ, ν, λ) selection replacing the carried-over ν/λ.
2. Wire the semantic ground cost: excavate K-space centroids via
   `compute_reduced_centroids`, aggregate to the 28 categories, sweep
   `blend_ground_costs(D_sem, D_temp, α)`.
3. Run the §17–§20 quality harness on `barycenter_rle_dba` outputs (decode to frames via
   `rle_decode` where the harness needs frame-level input) — now cheap at full fidelity.
4. Re-run the order-null (§15/§27 machinery) at the RLE scale, where the shuffle
   actually destroys order rather than run structure.
