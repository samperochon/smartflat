# ARC_AUDIT.md — Expert audit of the E→F→G barycenter arc (Kickoff L, 2026-07-03)

Evidence-backed review of everything the **E→F→G→(H/I/J/K)** barycenter arc added, across five
dimensions — **maintainability, accuracy, rigor, reusability, internal consistency** — plus a
phased, behavior-preserving refactoring plan. Linked from `RESULTS_HANDOFF_barycenters.md` §21.

**Audit surface.** `smartflat/features/symbolic_barycenter/{baselines, barycenter_quality,
order_evaluation, structure_metrics, vocab, visualization}.py`;
`smartflat/engine/distances/{_rtwe, _eshape_dtw, _alignment_paths, _bounding_matrix, _utils}.py`;
`tests/test_{order_evaluation, structure_metrics, barycenter_quality, barycenter_softdtw_ssg,
barycenter_msa_consensus, barycenter_discreteness_levers, baselines, distances_rtwe,
distances_eshape_dtw, barycenter_dba}.py`; notebooks `06f`–`06k` (read-only historical records).

**Method.** Read the 5 canonical docs (`coordination-EFG-roadmap.md`, RESULTS_HANDOFF §15–§20,
`BARYCENTER_METHOD_DISCRETENESS.md`, `PAPER_BRIDGE.md`, `CLAUDE.md`); mapped the code surface with
parallel `Explore` agents; verified every high-value finding firsthand (read the code, ran
repo-wide greps); cross-checked method↔citation faithfulness with a `literature-grounder`
(5/5 SUPPORTED). Every finding cites `file:line`. Adversarial verification downgraded two
candidate findings (a "dead code" flag and a "yardstick-not-common" worry) that turned out to be
false — noted inline.

**Severity.** **P0** correctness/leakage/repro-breaking · **P1** real debt · **P2** minor · **P3** nit.

## Session status (evidence)

- **Branch:** audit + Phase-0 fixes on `barycenter-arc-audit-L` (off `barycenter-quality-fgw`
  tip `ba43e62`). Arc commits: `78d7ef4` (E) → `e1cb674` (G) → `0b40eef` (§17) → `5d020f8`
  (§18) → `5feadc3` (§19) → `51354b9` (§20).
- **Frozen suites GREEN at session start:** `NUMBA_THREADING_LAYER=workqueue pytest` over the
  10 suites → **209 passed, 13 warnings in 392 s**.
- **Advertised §17–§20 public APIs all resolve** (imported by passing tests). The common rTWE
  yardstick `inertia_rtwe` is genuinely **fixed** at `nu=1e-4, lmbda=0.1`
  (`barycenter_quality.py:141,116`) — common across methods (a potential P0 ruled out).

---

## Headline verdict

The arc is **scientifically sound and unusually well-documented**. Every method is faithful to
its cited source (independently confirmed); the statistics are leakage-guarded and honestly
reported; the registry/harness design is genuinely reusable. The real debt is **structural and
consistency-level**, not correctness: one 2109-line module doing too much, and a `nu`/`lmbda`
default that drifted into **four** operating points across the arc's sessions. No P0 survived
verification.

| Dimension | Verdict | Worst item |
|---|---|---|
| Maintainability | Solid, one structural-debt item | **P1** `baselines.py` = 2109 lines, ≥6 jobs |
| Accuracy | Faithful (5/5 lit-confirmed) | **P2** eshape inner-cost hardcode (near-inert) |
| Rigor | Strong, honestly reported | **P2** fold-level bootstrap mildly anti-conservative |
| Reusability | Good, composable | **P2** `rtwe_alignment_path` arg-order footgun |
| Internal consistency | One real drift + DRY dup | **P1** `nu`/`lmbda` four operating points |

---

## Dimension 1 — Maintainability

**Verdict:** *Solid and well-documented, with one real structural-debt item.* Docstrings are
thorough and accurate; no repo-wide dead code; one stale TODO (now fixed). The genuine problem is
that `baselines.py` is the arc's junk drawer.

- **[P1] `baselines.py` is 2109 lines doing ≥6 jobs** — embed/decode, DTW/soft-DTW primitives,
  ~16 `barycenter_*` builders, 6 registry builders, CV helpers (`_make_clf`, `_nested_cv_auc`),
  evaluators (`evaluate_baselines`, `evaluate_incremental_ordering`, `baseline_significance_tests`,
  `bootstrap_*`). Every arc session appended here. *Fix (Phase 2):* split into `distances.py` +
  `builders.py` + `registries.py`, re-export from `baselines.py` for back-compat. *Note:* there is
  **no** configured linter (no ruff/pylint block in `pyproject.toml`), so the "too-many-lines
  warning" is not literally firing — the size itself is the signal.
- **[P2, dual accuracy] `_eshape_dtw_cost_matrix` hardcodes the inner rTWE cost** at
  `nu_rtwe=0.001, lmbda_rtwe=0.01` (`_eshape_dtw.py:161-162`), ignoring the `nu`/`lmbda` passed to
  `eshape_dtw_distance` (those drive only the *outer* edit penalties, `del_add=nu+lmbda` L167,
  `nu*|i−j|` L202). *Numerically near-inert* — the inner sequences are single columns `(1,1)`
  (L158-159) so stiffness/time barely bite — but it is an undocumented surprise and a **4th**
  distinct `nu/lmbda` operating point. Pre-dates the F arc (vendored). *Fix (Phase 1):* thread the
  passed values through or document the fixed inner cost.
- **[P2] `visualization.py` stale TODO + misplaced import — FIXED (Phase 0).** Removed the
  `plot_signals` `# TODO: 'info' was a notebook global` and moved the mid-file
  `utils_visualization` import to the header.
- **[P2] API-naming drift in builders** — `barycenter_dba_dtw(max_iters=…)` (plural,
  `baselines.py:191`) vs `max_iter` everywhere else; hand-rolled `barycenter_soft_dtw` (`:331`)
  lacks the `decode=` param its tslearn sibling `barycenter_softdtw` (`:368`) carries. *Fix
  (Phase 1):* rename with a back-compat alias.
- **[P3] Four functions have no in-file caller but are NOT dead** — `barycenter_mean_rtwe_dba`
  (`:890`), `pmatch_to_barycenter_stock_twe` (`:1083`), `dist_neg_pmatch_stock` (`:1109`),
  `ordinal_cost_matrix` (`:105`). Repo-wide grep: all four are exercised by `test_baselines.py`
  and used in `06b`/`06d` (mean-DBA-is-broken diagnostic + TWE ordinal-cost ablation). **Keep**
  (mention, don't delete). *(Adversarial note: an initial "dead code" flag here was a false
  positive — checking repo-wide, not just in-file, corrected it.)*

## Dimension 2 — Accuracy (scientific correctness)

**Verdict:** *Faithful — independently confirmed.* A `literature-grounder` cross-check returned
**5/5 SUPPORTED** with real sources: FGW α ↔ POT/Vayer (α weights the GW/structure term, α→0 =
feature/Wasserstein); Soft-DTW ↔ Cuturi-Blondel; SSG ↔ Schultz-Jain; MSA ↔ Gusfield /
Feng-Doolittle / Durbin; rTWE ↔ Marteau TWE (`nu`=stiffness, `lmbda`=edit penalty,
`del_add=nu+lmbda`, match penalty `2·nu·|i−j|`, `L_p`→`D_G[a,b]`). §17–§20 mechanism claims are
internally consistent with the numbers; no silent correctness bug found.

- MSA (`barycenter_msa_consensus`, `:774-887`): center-star medoid → pairwise rTWE align →
  "once-a-gap-always-a-gap" merge → Laplace-profile argmax → ≥50% occupancy gate. Deterministic;
  ties→lowest index; never emits empty. Faithful.
- FGW (`barycenter_fgw`, `:1234-1346`): `alpha` → `ot.gromov.fgw_barycenters`; α→0=Wasserstein
  corroborated by §17.3 (α=1 → freq_fidelity 0.19–0.26, entropy collapse). `dg`-decode
  `argmin_c (D_G@p)[c]` (onehot only; raises for mds). Faithful.
- Decode (`project_real_to_symbolic`, `:39-81`): `dg` = `argmin_c m[c]` = 1-medoid under `D_G`
  for a DBA mean (derivation correct; needs symmetric `D_G`, positive off-diagonal — satisfied by
  `build_g28_ground_cost`); honestly documented as a heuristic for optimized centroids.
- rTWE local cost (`_rtwe.py`): the four `precomputed_distances[a,b]` lookups mirror TWE; rolling
  and full-matrix kernels agree (26 passing tests). Prior `-2→-1` alignment off-by-one already
  fixed (PAPER_BRIDGE A1/A2), test-covered.
- **[P3] MSA "2-approximation" claim is conditional** — Gusfield's center-star 2-approx holds only
  for a **metric** ground cost. The docstring (`:787`) asserts it unconditionally. rTWE is
  TWE-derived (metric for a metric `D_G`, `nu,lmbda≥0`), so it likely transfers, but phrase it
  "2-approximation **under a metric ground cost**." Doc-only (Phase 3/4).

## Dimension 3 — Rigor (statistical / methodological)

**Verdict:** *Strong and honestly reported.* Leakage-guarded throughout; ΔAUC and BH designs
correct; nulls reported honestly. Two minor, disclosed caveats. *(Rests on firsthand reads of
every CV/bootstrap/permutation path; an independent skeptical-reviewer pass was dispatched but
interrupted by the session limit — a cheap optional re-confirmation, not a blocker.)*

- **No leakage:** `_make_clf` puts `StandardScaler` *inside* the logreg Pipeline (`:1915-1919`,
  refit per fold); RF is scaler-free; `_nested_cv_auc` (`:1929-1963`) fits `GridSearchCV` on the
  train fold only, scores the held-out fold; inner folds `max(2, min(5, min-class-count))`.
- **Paired folds / optimism cancellation (§15):** `order_information` enumerates the outer
  `RepeatedStratifiedKFold` splits **once** (`order_evaluation.py:201`) and reuses them for intact
  and every shuffle → ΔAUC cancels common-mode classifier optimism. Same one-enumeration pattern
  in `evaluate_incremental_structure` (`structure_metrics.py:721`) and
  `evaluate_structure_length_controlled` (`:811`).
- **Honest CI labeling:** §15's interval is a **permutation-null band**
  (`auc_intact − pct(null, 97.5/2.5)`, `order_evaluation.py:212-213`); the docstring says so and
  `order_helps = (ci_low>0)` is a valid one-sided test. The §15 *table* calls it "95% CI" loosely
  (P3 nit).
- **BH over the full family:** `structure_group_stats` corrects over the entire metric×comparison
  family jointly (`structure_metrics.py:664`; ~16 metrics × 3 = up to 48 tests), not per-comparison.
- **[P2] Fold-level bootstrap may be mildly anti-conservative:** §16's Δ(both−hist) CI resamples
  per-(repeat,fold) paired deltas as if i.i.d. (`structure_metrics.py:740-743`), but repeated-CV
  folds share subjects across repeats → variance can be underestimated. Doesn't overturn the sign
  (point deltas well inside CI); the width is optimistic. *Fix (Phase 4):* note the caveat, or move
  to a subject-level / block bootstrap.
- **[P3] Length standardized before CV** (`structure_metrics.py:798-799`, global mean/std incl.
  test folds) — inert for RF (scale-invariant) and re-standardized by the logreg pipe; worth a note.
- **[P3] No family-wise correction across §15's 15 cells** — acceptable: the verdict is all-null
  (0/15), and correcting a null only strengthens it.

## Dimension 4 — Reusability

**Verdict:** *Good — the registry contract and harness are genuinely composable*; shared helpers
are imported not re-copied (roadmap contract honored). Three small frictions.

- `{build, distance, kind}` is uniform; `score_barycenter_quality` auto-dispatches by output shape
  (sequence/histogram/matrix/medoid) with no per-method special-casing
  (`barycenter_quality.py:76-93,183-206`). `structure_metrics`/`order_evaluation` **import**
  `_nested_cv_auc`/`_make_clf`/`_rle` from the owners (no fourth CV/RLE copy) — roadmap §2/§4 met.
- **[P2] `symbolic_barycenter/__init__.py` re-exports nothing** (docstring-only) — the F-arc API is
  reachable only via full submodule paths; contrast `engine/distances/__init__.py` with a proper
  `__all__`. *Fix (Phase 3):* curated re-exports.
- **[P2] `rtwe_alignment_path` puts `precomputed_distances` 3rd-positional** (`_rtwe.py:424`) while
  `rtwe_distance`/`rtwe_cost_matrix`/`rtwe_alignment_path_with_costs` and `eshape_dtw_alignment_path`
  all put it last → callers pass it positionally in one place (`visualization.py:503`), by keyword
  elsewhere. Footgun. *Fix (Phase 3):* keyword-safe realignment or a prominent note.
- **[P2] Same base method, two native distances:** `barycenter_dba_dtw` is scored by `dist_dtw` in
  `default_baseline_methods` (`:1379`) but by `dist_rtwe` (native) in `discreteness_lever_methods`
  (`dba_dtw_dg`/`dba_dtw_cat`). The common `inertia_rtwe` yardstick is unaffected (fixed), but
  `inertia_native` for "dba_dtw" is **not comparable** across registries. Document, or standardize.

## Dimension 5 — Internal consistency (highest-yield axis)

**Verdict:** *One real convention-drift item (`nu`/`lmbda` defaults) plus DRY duplication in tests
and notebooks (both now addressed in Phase 0).* Committed numbers are safe (notebooks pin
`NU=1e-4, LMBDA=0.1` explicitly), but the mismatched **defaults** are a latent trap.

- **[P1] `nu`/`lmbda` default drift — four operating points for "the" rTWE cost:**
  - `0.001 / 1.0` — `default_baseline_methods` (`:1349`), `barycenter_mode_dba` (`:716`),
    `dist_rtwe` (`:1042`), `pmatch_to_barycenter`/`dist_neg_pmatch`, every public `_rtwe.py` default.
  - `1e-4 / 0.1` — `fgw_methods`/`softdtw_ssg_methods`/`msa_consensus_methods`/
    `discreteness_lever_methods`/`extra_experiment_methods`, `barycenter_mean_rtwe_dba`/
    `soft_mode_dba`/`msa_consensus`, `dist_eshape_dtw`, and the harness yardstick.
  - `0.001 / 0.01` — hardcoded inner cost in `_eshape_dtw_cost_matrix` (`:161-162`).
  - `1e-5 / 0.1` — documented as canonical in `PAPER_BRIDGE.md` §6 (thesis grid).
  A caller relying on defaults gets a *different* operating point per function. *Fix (Phase 1,
  gated by an oracle no-op proof):* a single `RTWE_NU=1e-4, RTWE_LMBDA=0.1` referenced by all
  higher layers; keep `_rtwe.py` kernel defaults; migration called out and proven a no-op on the
  committed §17–§20 numbers.
- **[P2] `FAMILY` taxonomy dict duplicated across notebooks — FIXED (Phase 0).** 06i/06j shared a
  literal (06j = 06i + `msa_consensus`), 06k a divergent copy. Added canonical
  `barycenter_quality.FAMILY` + `family_of()` (+ `tests/test_family_taxonomy.py`) for future
  notebooks to import. Historical notebooks left untouched.
- **[P2] Test fixtures copy-pasted — FIXED (Phase 0).** `_ground_cost` (×5), `_cohort` (×4),
  `_seq` (×2) were byte-identical across test files. Extracted to `tests/_bary_helpers.py`
  imported by the 7 files; call sites unchanged (`_ground_cost(6)` etc.), behavior byte-identical.
  *(Kept as arg-taking helper functions, not pytest fixtures, to avoid churning 40+ call sites and
  every test signature — a small, justified deviation from the plan's "conftest fixtures" wording.)*
- **[P3] `test_baselines.py` shadows conftest `D5`** (`:51`);
  `test_barycenter_discreteness_levers.py:26` re-sets a numba env var conftest already sets. Left
  as-is (harmless; out of Phase-0 scope).

## Cross-cutting — PAPER_BRIDGE reconciliation — DONE (Phase 0)

The F-arc quality roster (§17–§20) is **not** folded into PAPER_BRIDGE's 6-baseline Tier-A/B
tables. **Assessment: correct, not drift** — the roster answers *representation fidelity*, not
`tab:baselines` discrimination AUC. Added a **Tier D** subsection to `PAPER_BRIDGE.md` §1 stating
the roster is additive (not a replacement) and linking §17–§20 + `BARYCENTER_METHOD_DISCRETENESS.md`
+ this audit.

---

## Phased refactoring plan (additive, behavior-preserving)

Full plan: `.claude/plans/kickoff-l-squishy-meadow.md`. Frozen suite must stay green after every
step; no registry-key/builder-default change without an oracle-backed no-op proof.

- **Phase 0 (this session, on `barycenter-arc-audit-L`) — DONE:** conftest→`_bary_helpers` fixture
  extraction · `visualization.py` TODO+import cleanup · PAPER_BRIDGE Tier-D pointer · canonical
  `FAMILY`/`family_of()` + test · this `ARC_AUDIT.md` + RESULTS_HANDOFF §21 pointer.
- **Phase 1 (own branch) — `nu`/`lmbda` unification + naming:** module-level `RTWE_NU`/`RTWE_LMBDA`
  referenced by all higher layers; thread/document the eshape inner-cost hardcode; rename
  `max_iters`→`max_iter`. **Gate:** a frozen `quality_table` CSV oracle proving a no-op on §17–§20.
- **Phase 2 (own branch) — `baselines.py` split** into distances/builders/registries with
  back-compat re-exports; a public-surface test first; move file-by-file, suite green each step.
- **Phase 3 (own branch) — reusability polish:** curate `__init__` re-exports; realign
  `rtwe_alignment_path` arg order (keyword-safe); document the `dba_dtw` native-distance divergence.
- **Phase 4 (doc-first) — rigor caveats:** fold the bootstrap-anti-conservatism + length-standardization
  notes into §16 / docstrings; escalate to a subject-level bootstrap only if the paper needs it.

Phases 1–2 each merit a `.claude/prompts/kickoff-{M,N}-*.md` sub-session prompt; Phases 3–4 can
share a lighter clean-up sub-session.

## Verification strategy

1. **Frozen-suite gate** after every change: the 10 suites must stay **209 passed**.
2. **Numeric oracle** for Phases 1–2: freeze the L=128 `quality_table` for the merged registry to a
   CSV; assert byte-identical before/after (guards the §17–§20 numbers).
3. **Public-surface test** before the split: assert every `from …baselines import X` used by
   tests/notebooks still resolves post-split.
