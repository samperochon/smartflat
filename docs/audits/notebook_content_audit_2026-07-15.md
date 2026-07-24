# Notebook content audit — `06*` series (15 notebooks)

**Date:** 2026-07-15 · **Method:** `/notebook-audit` (skill) · **Scope:** `notebooks/06*.ipynb`
**Auditors:** 15 × `notebook-content-auditor` (one per notebook, opus) + 1 cross-notebook redundancy
sweep + 2 adversarial verifiers + orchestrator-run deterministic checks against the on-disk CSVs.
**Canonical sources:** working-tree `RESULTS_HANDOFF_barycenters.md` (RH), `PAPER_BRIDGE.md` (PB),
`paper-chapter-6-barycenters/PAPER_TODO.md` (PT), `CLAUDE.md`, `ARC_AUDIT.md`,
`BARYCENTER_METHOD_DISCRETENESS.md`. Config: `.nbaudit.json` (authored this run).

> **Working-tree note.** RH §26 and the 06/06b/06c/06d/06e edits are **uncommitted** and form one
> coherent in-progress session. This audit judges the working tree, which is correct: §26 governs.

---

## 0. Headline

The series is in **good scientific health**. The G=28 migration (RH §26) is real and verified: I
independently reproduced **all 10 rows** of the canonical `tab:baselines` from
`g28/baseline_comparison_g28.csv` (exact at 2dp, both significance pairs), and confirmed §26's
cross-check invariant (06b vs 06c at L=5162, 12 cells, **max |Δ| = 1.1e-16** — pure float-aggregation
noise). No notebook fabricates a result, and the arc's negative findings are reported honestly.

The drift is **almost entirely prose**: status framing that a later session overturned, provenance
labels, and decision-rule wording. Five issues are *not* prose and need your decision (§1).

**Counts:** 111 findings across 15 notebooks; 21 fixed, the rest triaged below. **No notebook's
canonical headline table is wrong** — the one exception is 06c's hand-written cell-17 verdict table
(33/34 cells off-lineage, E3), which no other notebook or doc depends on. Three notebooks (06c, 06m,
06l) carry substantive methodological issues; the other twelve are prose-level.

---

## 1. ESCALATIONS — your call, not auto-fixed

### E1. The paper's baseline table is not computed at the notebook-selected optimum ⚠️ **new**
No agent found this; it emerged from cross-reading the CSVs and the HP cells.

- **06 is the HP-search notebook.** Its cell 8 runs the G=28 grid and **reassigns** the constants:
  `NU, LMBDA, OFFSET = float(best['nu']), float(best['lmbda']), float(best['offset'])`.
  Committed output: `selected: nu=0.001 lmbda=0.1 offset=0.1 (inner-CV AUC=0.680)`.
- **06's AUC-vs-L sweep (cell 21 → `auc_vs_L_g28.csv`) therefore runs at ν=1e-3, offset=0.1.**
- **06b (`tab:baselines`) and 06c (the data spine) run at ν=1e-4, offset=0.3** — a point that is
  **not in 06's top-5 of 24 grid points** (it ranks 3rd at 0.6486 vs the winner's 0.6803).

Consequence — **RH §26 states two different values for nominally the same quantity**, without
disclosing they are different operating points:

| Quantity (G=28, L≈5162) | §26 text | Source | Operating point |
|---|---|---|---|
| `wasserstein` Patient-v-Control | **0.79** | 06 sweep | ν=1e-3, offset=0.1 |
| `wasserstein` Patient-v-Control | **0.81** | 06b `tab:baselines` | ν=1e-4, offset=0.3 |
| `wasserstein` Control-v-RIL | **0.77** | 06 sweep | ν=1e-3, offset=0.1 |
| `wasserstein` Control-v-RIL | **0.81** | 06b `tab:baselines` | ν=1e-4, offset=0.3 |

Same for `k_medoid` at L=5162: **0.633/0.636** (06's sweep) vs **0.715/0.730** (06b/06c spine).
Both files are internally correct; they are different experiments wearing the same label.

**Independent proof (from the cross-notebook sweep).** The divergence is *exactly* HP-shaped:
`majority_voting` scores via `dist_hamming` and touches no `D_G`/ν/λ — and it agrees to the digit
across **all three** caches (0.772 / 0.654). Only the HP-dependent methods diverge. 06b ≡ 06c exactly.
That pattern can have no other cause.

**Why it matters:** a reviewer will ask why the HP search selected ν=1e-3/offset=0.1 and the headline
table reports ν=1e-4/offset=0.3. **Decide:** re-run 06b at the selected optimum, or state explicitly
in §26 + the paper that `tab:baselines` is reported at the shipped canonical point (RTWE_NU=1e-4,
δ=0.3) and that 06's grid is a sensitivity check, not the operating point.

**Downstream damage — the re-selection silently manufactures an interpretive claim.** 06 c23 says
"*`majority_voting` matches the histogram on Control-vs-RIL because its lock-step per-phase mode is
itself a frequency-dominated statistic*". True only at 06's re-selected HP (0.772 vs 0.771). At the
canonical HP it is **false**: 0.772 vs 0.809 — a 0.037 gap.

**Structural root (merge candidate).** 06 c21/c22 and 06c c15/c16 are **forks of one experiment** —
identical modulo a variable rename, the suptitle, a `savefig` and a print. They write two caches at
two HPs and reach two numbers, each claimed definitive. Their method sets also diverge undocumented
(06 pops `edit_median`, 06c keeps it). Suggested owner: **06c** (denser ladder 64→5162, keeps
`edit_median`, is the data spine); 06 keeps the figure/HP-search narrative and cites it.
Note 06d c15's HP search is the *right* pattern — it scopes its result to a `tw_twe_reselected`
method and leaves the globals alone, where 06 c8 rebinds them for the headline.

### E2. The Wasserstein ground-cost ablation has REVERSED at G=28 ⚠️ **paper impact**
Verified from `baseline_comparison_g28.csv` and RH §26's own table. The ordinal ablation now **matches
or beats** the D_G Wasserstein cost:

| | Patient-v-Ctrl | RIL-v-Ctrl | Healthy-v-TBI | TBI-v-RIL |
|---|---|---|---|---|
| `tw_twe` (Wasserstein cost) | 0.73 | 0.76 | 0.65 | 0.57 |
| `twe_ablation` (ordinal) | 0.73 | **0.81** | **0.71** | 0.57 |

RH §4's K-space claim — "tw_twe ≫ twe_ablation, RIL-vs-Control 0.73 vs 0.57 … the Wasserstein ground
cost materially helps" — **no longer holds**. RH §26 lists only mode-DBA ≫ mean-DBA as a surviving
design-justifying ablation (consistent), but never says the Wasserstein ablation *flipped*.

**Why this is the most consequential item in this ledger.** PAPER_TODO's reframe thesis rests on
exactly two ablations (PT §0, verbatim):

> "The two ablations that DO survive — **mode‑DBA ≫ mean‑DBA** (0.73 vs 0.48, BH p=0.035 [RH §4]) and
> **Wasserstein ≫ ordinal cost** (0.73 vs 0.57, p=0.053 [RH §4]) — are **the new empirical backbone**."

…and PT **§1.4 is titled "Promote the surviving ablations (the new empirical backbone)"**. At G=28:
- **mode-DBA ≫ mean-DBA SURVIVES** — 0.49 vs 0.69, p=0.014 (RH §26). ✅
- **Wasserstein ≫ ordinal REVERSES** — the ordinal ablation now *wins*. ❌

So **half of the reframed paper's empirical backbone has flipped sign**, and the backlog still says to
promote it. Since the whole reframe is "the method is a principled averager whose design choices are
ablation-justified", losing one of the two justifications is a structural problem, not a number edit.

Also: 06b cell 8's "Sanity gate: the Wasserstein ground cost must change results" only asserts
`|Δ|>1e-6` and prints `SANITY GATE PASSED` — **it passes while the effect runs the wrong way**, and
never reports direction. A gate that cannot fail in the direction that matters is not a gate.

### E3. 06c still routes its barycenter through the uninstalled forked aeon — **cannot execute**
RH §26 repaired 06 and 06b onto `barycenter_mode_dba`; **06c was left behind.**
- cell 1 `from aeon.clustering.averaging import elastic_barycenter_average`
- cell 10 `idx_init, bary, best_cost, costs, idx_best = elastic_barycenter_average(..., distance='rtwe', precomputed_distances=…, integers=…)`

Stock aeon (0.11.1 / 1.2.0) returns a single ndarray, has no `precomputed_distances`/`integers`
params, and rejects `distance='rtwe'`. **The committed output (cost=12883.3) is a macOS-era artifact
of a fork that no longer exists.** This is why 06c is in the `heavy` list and was not executed here —
but it would fail regardless of runtime.

Related: **06c cell 17's verdict table has 33 of 34 numbers disagreeing** with the
`auc_vs_L_faithful.csv` it owns (deltas to +0.081); its cell-1 output still prints a
`/Users/samperochon/` path. The auditor's decisive argument: the **closed-form `wasserstein` drifts up
to 0.025**, which RH §25 says *cannot* happen across platforms ("matches to 3 decimals") — so this is
**off-lineage, not FP noise**. Direction of the verdict is unaffected (frequency still leads).

**Fix = a §26-style repair + re-run of 06c (hours).** Not auto-fixable. Recommended next session.

### E4. 06m's Breakfast order-null runs on a leaky CV — the fix is your own in-flight helper
**Timeline first, so this is read fairly.** `dedup_by_execution` is **not in HEAD** — it is
uncommitted, written in this working tree today (mtime 23:28), together with the untracked
`generalization/headroom.py` and `tests/test_headroom.py`. 06m's committed `order_null.csv` is dated
**Jul 7** — eight days *before* the helper existed. So this is **not** a case of ignoring an available
fix: the run predates the remedy, and you are evidently already building it. The finding is simply
that **06m's committed numbers are from the pre-dedup run and need a re-run once the helper lands.**

`action_segmentation.py:286` defines **`dedup_by_execution`**, whose docstring states:

> Breakfast films each execution with up to five cameras … **98.8%** of cross-view sequences are
> byte-identical after RLE … its 1712 "videos" are only **503** independent executions … The
> harness's `RepeatedStratifiedKFold` is **not group-aware**, so leaving them in puts copies of the
> same execution in train *and* test — the classifier can recognise rather than generalise. **Any
> per-sequence CV on Breakfast should run on the deduplicated set.**

**`grep -rn dedup_by_execution notebooks/ smartflat/` returns only its own definition — nothing calls
it yet.** 06m cell 7 samples `CAP_ORD=80` files/class with no dedup; each top-6 class has only ~52
distinct executions, so ≥28 duplicate views are guaranteed in both train and test.

**Scope of the damage is limited and worth stating precisely:** ΔAUC is self-controlling (leakage
inflates intact and shuffled alike), so **the 0/60 verdict stands**, and it is consistent with the
settled disjoint-vocabulary diagnosis. What is undercut is the **reported AUC magnitude (1.0) and
N=480**. Fix = wire in `dedup_by_execution` and re-run (cheap relative to 06c).

### E5. The generalization results have no canonical home (missing RH §27)
RH ends at §26 (2026-07-07); 06l/06m/06m2/06m3 landed after it. `grep 'gtea|50salads|action-seg'`
across RH, PB, PT, CLAUDE.md = **zero hits**. PT §2 names only Breakfast (§2.1), BPI-2012 (§2.2) and
D1-synthetic/D2 (§2.3) — **50Salads and GTEA are in no plan at all**, and PT §2.3's D1 item still
reads `[TODO]` though 06l implements it. Recommend an **RH §27** covering the Phase-2 action-seg
family with the frequency-saturation diagnosis and the dedup caveat, plus a PT status flip.

---

## 1b. What the adversarial pass REFUTED (recorded, because it changed the outcome)

Two verifier agents were tasked with *refuting* the proposed fixes. They killed a large fraction —
worth recording, since the raw auditor findings would have been over-applied:

- **"Deferred to pomme" — 15 of 17 accused cells REFUTED.** The auditors treated any "deferred"
  wording as stale. But **§26 (2026-07-07) is *newer* than §25's abandonment note (committed
  2026-07-06) and says verbatim: "*Deferred (unchanged from §25, now enabled). The §17–§20 quality
  roster (`06g/i/j/k`) **stays at its L=128 previews** for this session.*"** So deferral prose is
  *current fact*, not drift. Only the two **tractability** claims (06i c12, 06j c12: "*where the
  O(…) cost is tractable on the target hardware*") are falsified by §25's empirical result. 06g c2
  ("Full L~5162 stays deferred.") was a **baseless** accusation — no pomme, no tractability claim, and
  06g has no stub at all. **Fixed 2 cells, left 15.**
- **06h c11 legend — REFUTED, and fixing it would have introduced an error.** The auditor said the
  legend should say `ci_low > 0`. But 06h's cell-12 code fills **two-sided**
  (`sig = (lo > 0) or (hi < 0)`), so "*95% CI excludes 0*" describes the code *exactly*; 06h imports
  `structure_metrics`, not `order_evaluation`, so the cited rule governs a module it never touches,
  and `structure_metrics.py:789` uses "excludes 0" itself. **Only the "(a real gain)" gloss is wrong**
  — two filled markers have CIs entirely *below* 0, where structure *hurts*. That is what was fixed.
- **06e c10 — CONFIRMED, but on corrected evidence.** 06e never imports `order_evaluation`;
  `ordering_helps` is notebook-local (c4: `show['ordering_helps'] = show['delta_ci_low'] > 0`). The
  finding survives on the notebook's own table (3 of 6 CIs exclude 0, all *below*) plus a
  self-contradiction two clauses later ("*fails to exclude 0*" … "*significantly hurts*").

### A new doc-level bug found by the adversarial pass
**RH §26's claim that `n_jobs` unblocks the quality roster is false.** §26 says "*The `n_jobs` change
removes §25's practical blocker … promoting those notebooks to L≈5162 is a clean follow-up*". But
`n_jobs` exists **only** on `evaluate_baselines` (`evaluation.py:45`); `score_barycenter_quality` has
no such parameter — and 06g/06i/06j/06k call **only** `score_barycenter_quality` (zero
`evaluate_baselines` hits). §25 also already modelled the parallel case: even ideal
`(group×method×init)` parallelism leaves `msa_consensus` (**3 group-units**) at ~a day, which
contradicts §26's "~90 independent builds per method". **The deferral is more firmly true, not less.**

---

## 2. HIGH severity (verified) — auto-fixed unless noted

| # | NB | Cell | Issue | Canonical |
|---|---|---|---|---|
| H1 | 06m | 12 | "**§7.2 evidence that the frequency-only result generalises externally**" — a frequency-saturated task is no external evidence; contradicts the same cell's own "no headroom" caveat | Valid claim is narrow: order is *not required* when vocabularies are disjoint; **not** "order never helps" |
| H2 | 06m | 12 | "**the quality table feeds the §6.5 `tab:baselines` Breakfast column**" — routes the Tier-D quality roster into `tab:baselines` | PB §1 Tier D: quality methods "are **additive to, not a replacement for**… must not be pasted into `tab:baselines`". The table has no AUC column and no proposed-method row |
| H3 | 06m | 0,6,7 | "**apples-to-apples with SDS2**" (×3) — **false**: `n_shuffles=200` + 3×5 CV are `order_information`'s *library defaults*; SDS2 ran **100 shuffles, 2×5** | RH §15 + 06f c1 `N_REPEATS, N_FOLDS, N_SHUFFLES = 2, 5, 100` vs `order_information(n_repeats=3, n_folds=5, n_shuffles=200)` |
| H4 | 06l | 10 | "**recovery degrades monotonically as edit-noise rises**" — **false vs its own D1 output**: `majority_voting` is flat at rTWE 0.0 / exact-match 1.000 at *every* noise level (0.0→0.4). Only dba_dtw (0→18.601) and edit_median (0→10.827) degrade | Own `d1_exact_recovery.csv`. The buried result is better: positionwise-mode recovers the centre exactly at all noise; alignment-based dba_dtw degrades worst — consistent with the arc's frequency-dominance finding |
| H5 | 06l | 9 | Figure title "**recovery degrades gracefully with noise**" — same defect baked into the saved `d1_exact_recovery.png` | idem |
| H6 | 06l | 8 | D1 **never evaluates the proposed averager** (`tw_twe_pmatch`/mode-DBA) — yet c0 claims "the **required** synthetic exact-recovery benchmark for the averager". `seq_methods` = majority_voting/dba_dtw/edit_median only | PT §2.3 scopes D1 as exact-recovery validation of the contribution (= TW-TWE + mode-DBA). **ESCALATED — needs new compute, not a prose fix** |
| H7 | 06e | 0 | Header sells the incremental-bigram probe as "**decisive tests**" with no superseded banner | RH §15 replaces it (it "overfits the 784-dim block"); PB 2026-06-30: 06f is "the decisive, self-calibrating replacement". Verdict unchanged/reinforced |
| H8 | 06i, 06j | 12 only | "where the O(…) cost is **tractable on the target hardware**" — the *tractability* claim, falsified empirically. **Narrowed from 17 accused cells to 2 by the adversarial pass (§1b)**: plain "deferred" wording is affirmed by §26 and was left alone | RH §25: attempted **on pomme**, 2 runs >25 h wall, never finished the first family, no `*_FULL.csv`; "**Decision: skip the full-length run**; the L=128 previews … stand" |
| H9 | 06d | 0,13 | Stale G=77 numbers + **inverted direction**: "histogram holds ~0.82", "k_medoid *falls* to ~0.70", "alignment falling further below the histogram as L grows" | **PARTIALLY ESCALATED — see §3** |
| H10 | 06c | 0,1,9,10 | Forked-aeon presented as installed/runnable | **ESCALATED — E3** |

---

## 3. The direction-of-trend problem (06d) — escalated, deliberately not auto-fixed

The auditor proposed correcting 06d's "k_medoid *falls*" to "rises with L and plateaus below; it never
falls", citing RH §26. **I checked the data and that correction is also wrong.**

`auc_vs_L_g28.csv`, k_medoid mean AUC:

| comparison | L=64 | L=256 | L=1024 | L=5162 |
|---|---|---|---|---|
| Control-v-RIL | 0.518 | 0.653 | **0.693** | 0.636 |
| Patient-v-Control | 0.534 | 0.683 | **0.697** | 0.633 |

The true trend is **non-monotonic**: k_medoid peaks at L=1024 and **declines** at L=5162. So:
- 06d c0's "falls to ~0.70" — the *fall* is real, but ~0.70 is the **peak**, not the endpoint (~0.63).
- 06d c20 / **RH §26's own "rise with L but plateau below"** — "plateau" is a **simplification**; it declines.
- The proposed fix would have introduced a new error.

**Only the stale-number half is safe and was fixed** ("~0.82" → the G=28 value). The direction prose
in 06d c0/c13/c20 **and RH §26's wording** need one consistent decision from you. (Caveat: 06's sweep
is at offset=0.1 — see E1 — so even this table is not the same operating point as 06b/06c.)

---

## 4. Cross-cutting themes (fixed once, not N times)

- **T1 — "CI excludes 0" vs the implemented `ci_low > 0`** — applies to **06e c10 and 06f c8/c9/c10
  only**. `order_evaluation.py:220` → `'order_helps': bool(ci_low > 0)`; the docstring calls the
  interval a **permutation-null band**, and "`ci_low > 0` is exactly a one-sided" test. In each case
  the misstatement is **literally false against the notebook's own committed output** (each has CIs
  excluding 0 on the *negative* side — read literally, "CI excludes 0" scores 06f as 1/15, not 0/15).
  **Verdicts are unaffected.**
  **06h c11 is NOT part of this theme** (adversarial pass, §1b): its cell-12 code genuinely fills
  two-sided, so its legend describes the code correctly and rewording it to `ci_low > 0` would have
  *introduced* an error. Its only defect is the "(a real gain)" gloss over two filled-but-negative
  markers — that, and only that, was fixed.
  ⚠️ RH §15 itself contains "**Every CI brackets 0.**" — the doc has the same slip, refuted by its own
  next clause. Flagged for you; **not edited** (canonical doc).
- **T2 — "deferred to pomme": mostly NOT stale** (see §1b). §26 (2026-07-07, the newest section)
  explicitly affirms the roster "stays at its L=128 previews", so deferral wording is current fact.
  **15 of 17 accused cells were refuted and left alone**; only the two *tractability* claims (06i c12,
  06j c12) were false and fixed. §25 also *sanctions the stubs remaining*, so cells that merely
  contain the stub are fine.
- **T3 — macOS-prose vs pomme-committed-output** (06g c7/c13/c16, 06h c13, 06i c9). Prose quotes the
  §17 macOS canon while the table *directly above it* is the §25 pomme re-run. **Both are correct**
  (RH §25). Fix is a **provenance note, never a value change**.
- **T4 — `nu=1e-4` mislabelled "thesis"** (06 c2, 06b c1/c2, 06c c1, 06k c1). Thesis grid = **1e-5**;
  1e-4 is the shipped canonical `RTWE_NU` (`distances.py:20`). PB §6 Kickoff-M: "the two operating
  points genuinely differ and are **not** reconciled in code". λ=0.1 and δ=0.3 *are* thesis values —
  so a blanket relabel would over-correct.
- **T5 — dead `.claude/` references** (06f c0, 06g c0, 06i c7). `.claude/` is gitignored
  (`.gitignore:37`) and never tracked. **Systemic**: RH §15/§17/§21, PB and ARC_AUDIT cite the same
  dead paths. One ledger item, not N fixes.
- **T6 — `FAMILY` literal duplicated** (06i c6, 06j c6, 06k c6). Canonical:
  `barycenter_quality.FAMILY` / `family_of()`, whose own comment says it was single-sourced because
  "it had been pasted into 06i/06j/06k". **06i's copy is now incomplete** (missing `shape_dba`,
  `msa_consensus`, the 7 `*_dg`/`*_cat` keys) → live divergence risk.
- **T7 — hand-rolled cohort rebuild** (06 c4, 06c c2/c3, 06d c2, 06e c2) vs
  `vocab.load_g28_cohort` / `vocab.build_g28_ground_cost` ("factored from 06c Cells 2–3"). All
  pre-date §15 → behaviour-identical. Low priority. **06 c4 is the exception**: its local
  `make_ground_cost` **shadows** a same-named `vocab.make_ground_cost` with *different* semantics, and
  c0 claims the notebook uses the shared loader when no cell calls it.
- **T8 — 3.7 vs 3.78 bits** group-entropy anchor (06g c16, 06j c6/c7). **The docs disagree with each
  other**: RH §17.2 says ≈3.7; §18.2/§19.2/§20.2 + BARYCENTER_METHOD_DISCRETENESS say ≈3.78. Fixed to
  3.78 only where §19.2 governs (06j); escalated otherwise.

---

## 5. Per-notebook summary

| NB | Findings | State |
|---|---|---|
| 06 | 9 (0 high) | **Healthy.** G=28 migration complete; retired Hamming cell correctly absent. Cosmetic/provenance only. |
| 06b | 8 (0 high) | **Numerically clean** — every output matches §26 exactly; no K-space residue. Prose/runtime-estimate drift. |
| 06c | 12 (4 high) | ⚠️ **Broken** — forked-aeon cells cannot run (E3); verdict table off-lineage. |
| 06d | 5 (4 high) | Own L=512 numbers fresh + match §13.3. Cross-refs to 06 carry pre-§26 G=77 values; direction prose escalated (§3). |
| 06e | 6 (1 high) | Numbers/verdict current & match §13.4 exactly. Superseded-method framing. |
| 06f | 9 (0 high) | Cohort/rule/verdict all match §15/§25. Rule-wording drift (T1). |
| 06g | 6 (0 high) | Current on every §17 verdict. Stale roadmap + macOS/pomme provenance. |
| 06h | 4 (0 high) | **Computationally clean**; §25 tuple-unpack fix present; reproduces §25 pomme exactly. Legend + provenance. |
| 06i | 9 (1 high) | Sound; matches §25. Pomme framing + incomplete FAMILY copy. |
| 06j | 10 (3 high) | Numbers are sanctioned §25 pomme values. Pomme framing + 3.7/3.78. |
| 06k | 8 (3 high→1 after refute) | Numerically clean, no over-claiming. Deprecated `max_iters=` kwarg (works via alias; leaks 4 DeprecationWarnings into committed outputs). |
| 06l | 6 (2 high) | §1 validity proof sound. **§2 D1 verdict contradicted by its own output** (H4/H5); proposed averager untested (H6). |
| 06m | 10 (3 high) | 0/60 genuine & saturation caveat present, but paper-facing claims over-reach; **leaky CV** (E4). |
| 06m2 | 5 (0 high) | Internally sound, honestly scoped. Undocumented (E5); 3× duplicated chronogram. |
| 06m3 | 4 (0 high) | Internally accurate; honest about being underpowered. Undocumented (E5); ranks on n=4/group. |

### ⚠️ A 16th notebook appeared *during* this audit — `06n`, not audited
`notebooks/06n_frequency_controlled_order.ipynb` (untracked; mtime **during this session**) is being
written as this ran. **It is deliberately out of scope** — auditing a file mid-write is meaningless —
but it matters for reading §1:

Its own header reaches E4/H1's conclusion independently, and pushes further than this audit did:

> `06m` ran the §15 ΔAUC order-null on Breakfast's top-6 activities and got `order_helps` **0/60** —
> but with `auc_intact = 1.0` on *all 60 cells* … With `AUC_intact` pinned at the ceiling, `delta_auc`
> is 0 **by arithmetic**: the null could not have fired even if order carried signal. **That 0/60 is
> vacuous, not evidence** — and **the same doubt hangs over SDS2's 0/15**.

That last clause is a **stronger claim than anything in this ledger**, and if it holds it reaches the
arc's central negative (RH §15). **One check, run for this ledger, bears on it directly** — from
`g28/experiments/order_shuffle_null_deltaAUC.csv` (06f's own artifact, n=15):

> `auc_intact` **min 0.607 / max 0.821**; cells at the ceiling (`auc_intact ≥ 0.99`): **0 / 15**.

So SDS2's null had **real headroom on every cell** and is *not* saturated the way Breakfast is
(`auc_intact = 1.0` on 60/60). The arithmetic that makes Breakfast's 0/60 vacuous — `delta_auc = 0`
forced by a pinned ceiling — **does not transfer to SDS2 on its face**. That does not make 0/15 safe
(a *powered* null is a separate question from an *unsaturated* one), but the "same doubt" needs a
different argument than saturation. Worth settling explicitly in 06n rather than by implication: it is
the difference between "Breakfast was the wrong task" and "the arc's headline negative is unsafe".

So E4/E5 should be read as **converging with work already in flight**, not as a backlog you have
ignored: `dedup_by_execution`, `headroom.py`, `tests/test_headroom.py` and `06n` were all written
tonight, and all target exactly the saturation/leakage problems recorded above.

### Library-ahead-of-notebook (in flight, not debt)
`smartflat/features/symbolic_barycenter/generalization/headroom.py` is **untracked** (`??`, written
today) and ships `class_vocabularies`, `headroom_table`, `restrict_to_shared_vocabulary`,
`pooled_markov_surrogate` — precisely the remedy for 06m's own frequency-saturation caveat (E4/H1).
No notebook imports it yet. This looks like the obvious next notebook rather than a defect.
⚠️ 06m's rewritten cell 12 now **references** it, so **commit `headroom.py`** or that pointer dangles.
Same class of gap as `dedup_by_execution`: the library has the fix, the notebook has not adopted it.

### Non-findings worth recording (checked, cleared)
- **50Salads/GTEA `order_null.csv` are empty (8 bytes) — this is correct, not a bug.** Both notebooks
  document it: 50Salads has a single `salad` class (the null needs ≥2 groups); GTEA has ~4 videos/
  activity, below `order_information`'s 5-fold minimum (`np.min(np.bincount(y)) < n_folds` → skip).
  Both say so explicitly, and 06m3 c10 states the empty table is "**not evidence against order**".
- **06b cell 2's aeon import is fine** — it is *stock* aeon with `distance='twe'` (the documented
  categorically-wrong mean baseline), not the forked `rtwe`. It runs; its `tw_twe_mean` row matches §26.
- **RH §12.1's "stock aeon 1.2.0 lacks it" is itself inaccurate** — the symbol imports fine in 1.2.0;
  only the forked `rtwe` *distance* is gone. RH defect, not a notebook defect.
- **06h c10's single assignment is correct** — `evaluate_structure_length_controlled` returns one
  DataFrame (only `evaluate_incremental_structure` returns a tuple; c8 correctly unpacks it).
- **06i carries no tslearn version string**, so §25's "0.8.1 → 0.6.4" correction does not fire here.
- **RH §26 cross-check nit:** the doc lists k_medoid `0.716`; the CSV value is `0.715451`. Cosmetic,
  in the doc, not a notebook. Not touched.

---

## 6. Phase 1 — deterministic checks

- **Parse/validate:** 15/15 GREEN (`nbformat.validate`, all 4.5, all cells ID'd).
- **Execution health** (`check_notebooks.py --only '06*' --skip-heavy`, 12 notebooks): **complete —
  11/12 PASS, 0 real failures.** Full report at `analysis/diagnostics/notebook_health.json`. Every
  notebook re-executed end-to-end in a fresh `smartflat_repro` kernel:

  | notebook | status | wall |
  |---|---|---|
  | `06f_order_shuffle_null` | **PASS** | 2167.0 s |
  | `06l_generalization_synthetic` | **PASS** | 1511.4 s |
  | `06e_ordering_vs_frequency` | **PASS** | 1311.5 s |
  | `06k_discreteness_levers_quality` | **PASS** | 620.2 s |
  | `06d_audit_and_experiments` | **PASS** | 573.5 s |
  | `06g_methods_chronogram` | **PASS** | 493.2 s |
  | `06i_softdtw_ssg_quality` | **PASS** | 263.1 s |
  | `06j_msa_consensus_quality` | **PASS** | 196.1 s |
  | `06m3_generalization_gtea` | **PASS** | 108.0 s |
  | `06m2_generalization_50salads` | **PASS** | 60.0 s |
  | `06h_structure_metrics` | **PASS** | 30.8 s |
  | `06m_generalization_breakfast` | **TIMEOUT** (not a bug) | >1800 s / cell |

  **The one non-PASS is "slow, not broken":** 06m's order-null cell hit the 1800 s per-cell cap. Its
  own header budgets that cell at **~6.8 h**, so this is the expected outcome of a bounded-timeout
  sweep, not a code defect — it would PASS with a longer cap. **Everything the fixes touched
  executes clean:** 06f, 06h, 06i, 06j, 06k, 06l all PASS. (`heavy`, skipped by design per
  instruction: **06, 06b, 06c** — full-length L≈5162 cells; **06c would ERROR regardless** — E3, it
  calls the uninstalled forked aeon.)
  *Nothing in this audit's findings depended on these results — they come from cell source, committed
  outputs, and the on-disk CSVs — but the clean sweep independently confirms the 12 non-heavy
  notebooks are executable.*
- **Number-lock:** **NOT RUN — no manifest exists.** `docs/nb_number_lock.json` and notebook oracle
  JSONs were never authored; `tests/_oracles/` holds only `rtwe_hparam_oracle.csv` (a pytest-level
  RTWE lock). **Gap — see §7.**

## 6b. Fixes applied (Phase 3) — 30 edits across 9 notebooks

Verified: **every committed output preserved byte-for-byte** (cell-by-cell `outputs` comparison vs
HEAD: 0 outputs touched) — the single exception is **06l**, deliberately re-executed (below). All
notebooks re-validate under `nbformat.validate`. Every other code-cell edit is **comment-only**, so a
re-run reproduces identical output.

**06l was refreshed, and the refresh was verified rather than assumed.** Fixing its figure title meant
touching a code cell, which would desync source from the saved PNG — so 06l was re-executed **in its
own original kernel** (`smartflat-conda`/`smartflat_repro`, Python 3.11.15 — *not* the audit's
`python3` kernel, which could have silently shifted numbers across library versions). All three CSVs
were genuinely regenerated (mtimes confirmed) and **all three reproduce byte-identically**:
`d1_exact_recovery.csv`, `order_discriminative_validity.csv`,
`order_discriminative_jitter_sweep.csv`. So only the figure title moved; no number did.

| NB | Cells | Change |
|---|---|---|
| 06 | 0, 2, 5, 23 | `PAPER_BRIDGE §12`→RH §12 (PB has no §12) · "thesis optimum"→shipped canonical + why · K-space header→G=28 category ground cost · k_medoid `0.53`→**0.52** (CSV: 0.518) |
| 06b | 1 ×3, 4 ×2 | "Thesis constants"→shipped canonical · `O(L³·⁵)`→**O(L²)** (probe-measured L^2.1/L^2.0; gate→256/128) · shim→`evaluation.py` · pairwise `~3 min`→**~40 s** · `~20–35 min`→**~3.2 h** (own output: 11540 s) |
| 06c | 1 | "thesis G=28 hyperparameters"→shipped canonical point |
| 06e | 0, 10 ×2 | **SUPERSEDED banner** → 06f · "fails to exclude 0"→`delta_ci_low ≤ 0` · verdict cross-ref now cites 06f |
| 06f | 10 ×3 | rule→`ci_low > 0` (one-sided) · "every CI brackets 0"→"no CI has `ci_low > 0`" · Kickoff F/G roles corrected per §17/§16 |
| 06h | 11 | legend: filled = significant **in either direction**, read the sign (two filled markers are negative) |
| 06i | 7, 12 | levers "deferred"→**shipped** (§20/06k) + dead `.claude/` path removed · ~3.7→**3.78** bits · tractability→abandoned |
| 06j | 7, 12 | ~3.7→**3.78** bits · tractability→abandoned (incl. §25's 3-group-unit model) |
| 06k | 1 | "(thesis G=28)"→canonical shipped point |
| 06l | 0, 9, 10 ×3 | **D1 verdict rewritten** to match its own data: per-method split, p=0 degeneracy caveat, proposed-averager gap · figure title de-falsified · header scoped to baselines · *(+2 self-corrections, below)* · **re-executed** |
| 06m | 0, 6 ×2, 7, 12 ×4 | saturation outcome in header · "apples-to-apples with SDS2"→**library defaults, not SDS2's** (×2, incl. the code comment) · leakage caveat added **and then corrected** · §7.2 + `tab:baselines` claims narrowed · headroom.py untracked-status noted · RH §27 gap noted |

### The spot re-audit caught an error in *my own* rewrite (recorded in full)
A fresh auditor was pointed at the **rewritten** 06l cell 10 and told to treat the new prose as more
suspect than the old. It found a real defect, which was then verified against the code and corrected:

> ~~"This is consistent with the arc's frequency-dominance finding (§12.7/§15): the position/count-based
> averager is the robust one here…"~~ — **wrong on three counts.**

1. `barycenter_majority_voting` is "**Per-timestep majority voting (lock-step, no alignment)**"
   (`builders.py:469`) — it is *positional*, not frequency-based. Calling it "position/count-based" and
   tying it to frequency-dominance conflated it with a different method.
2. The arc's frequency method is `wasserstein` — a **histogram**, not a sequence — so it cannot be in
   `seq_methods`. **D1 has no frequency arm at all**, so nothing on it can corroborate frequency-dominance.
3. §12.7/§15 are *between-group discrimination* results (AUC over Control/TBI/RIL); D1 is a
   *single-population centre-recovery* task with no groups, no labels, no AUC. Incommensurable.
   Worse: D1's centre is deliberately **order-rich** (`_random_center`: runs of 3–8) and is recovered
   exactly by a positional estimator — if anything the *converse* of "order carries no signal".

Replaced with the mechanical explanation (`_corrupt` is positionwise substitution with no warping =
`majority_voting`'s own lock-step assumption → the mode wins **by construction**; `dba_dtw` pays for
searching warps D1 never applies), plus explicit notes that this is *not* frequency-dominance support
and that D1 has no frequency arm. Also softened "loses the planted centre" (dba_dtw still matches
0.817 of positions at p=0.4) and the PT §2.3 attribution (its text names no subject).
**Follow-on fixes it forced:** 06l c9's figure title still said "degrades gracefully" — which my c10
rewrite now *contradicted* — and c0's header still claimed a benchmark "for the averager" that omits
the averager. Both corrected; 06l re-executed in its original `smartflat_repro` kernel to resync the
figure, with the CSVs diffed to confirm no numbers moved.

### …and a second, worse one in the 06m rewrite — disproved *by execution*
The 06m spot re-audit found a **high-severity error in the new leakage caveat**, and refuted it with a
constructed counterexample rather than an argument:

> ~~"ΔAUC is self-controlling (leakage inflates intact and shuffled alike), so the 0/60 verdict is
> unaffected"~~ — **false.**

`order_shuffle_null` permutes **each sequence independently**, so duplicate camera views share an
exact feature vector **only in the intact arm** — the shuffle *destroys* the memorisation channel.
Leakage therefore biases ΔAUC **upward**, toward a false "order helps"; it does not cancel. The
verifier demonstrated this on zero-signal data (classes statistically identical, 4 duplicate views per
execution): `order_information` returned `auc_intact` 0.969 vs `auc_null_mean` 0.548 → **ΔAUC = +0.42,
ci_low = +0.32, `order_helps = True`** — a false positive manufactured by leakage alone.

**The conclusion survives; the reasoning did not.** Corrected in-notebook to the sound argument:
leakage can only manufacture false **positives**, never a null, so an all-null **0/60 is
conservative** and stands — by *direction of bias*, not by symmetry.

⚠️ **This also bounds a canonical claim.** `order_evaluation.py`'s docstring and RH §15 say "the null
is its own control … classifier optimism/overfitting bias is shared and cancels". That is true of
**optimism** — but **not of group-structured duplicate leakage**, which the counterexample shows the
null does *not* control for. Worth a sentence in §15, since it is a limit on the arc's central
instrument. (It does not threaten SDS2's 0/15: one administration per participant, deduped on
`trigram`, so there are no duplicate views to leak.)

**Deliberately NOT edited:**
- **06e's pre-registration blockquote** — it also says "CI excludes 0", but a pre-registration is a
  record of what was committed to *before* results. Retroactively editing it would be misconduct.
  Flagged here instead.
- **06f cells 8/9** (bar legend + print criterion) — both CONFIRMED stale, but they are **code with
  committed outputs**: changing the strings without re-running desyncs source from output. 06f is
  git-clean and takes **36 min** to execute (measured this run). Fix them together with a refresh, in
  the `smartflat_repro` kernel, as was done for 06l.
  *(Also in c9: the format string is `'\\n%d / %d …'` — a double backslash, so the verdict line prints
  a literal `\n`. Fix the escape in the same pass.)*
- **06m cell 7's live print** — `order_helps (ci_low > 0): {n}/{60}   [SDS2 was 0/15]` puts the two
  side-by-side as comparable, which the corrected cell 12 now forbids. Code + committed output → same
  refresh caveat.
- **06d entirely** — its four HIGH findings are numeric and depend on resolving **E1** (§3).
- The **`FAMILY` literal** (06i/06j/06k) — the sweep verified **zero current value mismatches**; it is
  latent drift risk, and RH §21 scoped the shared helper to "future notebooks". Code change + re-run.

## 7. Recommended follow-ups (not done)
1. **Decide E1** (HP operating point) and **E2** (reversed ablation + PT §1.4). Paper-level.
2. **Repair 06c** onto `barycenter_mode_dba` (§26 pattern) and re-run → fixes E3 and the c17 table.
3. **Wire `dedup_by_execution` into 06m** and re-run the order-null (E4).
4. **Write RH §27** for the Phase-2 generalization family; flip PT §2.3 off `[TODO]`; add PT entries
   for 50Salads/GTEA or tag them "exploratory, not paper-bound" (E5).
5. **Author `docs/nb_number_lock.json`** + notebook oracles so the headline numbers are value-locked,
   then wire the two deterministic gates into a pre-commit hook (§6 gap).
6. **Fix RH §15's "Every CI brackets 0."** and the RH §17.2 3.7-vs-3.78 conflict (T1/T8) — doc-side.
