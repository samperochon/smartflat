# Frequency-Controlled Order Test — Results of Record

> **Does symbol *order* ever carry class signal beyond symbol *frequency*, on real data?**
> Self-contained record of the frequency-headroom screen + frequency-controlled order test.
> Written as a drop-in **§27** for `RESULTS_HANDOFF_barycenters.md` (kept separate only
> because that file is currently uncommitted WIP — fold in when convenient).

- **Date:** 2026-07-16 · **Branch:** `barycenter-generalization-datasets` · **Commit:** `82e4b4d` (on `8f9d0fd`)
- **Notebook:** `notebooks/06n_frequency_controlled_order.ipynb` (executed headless, 0 errors)
- **Module:** `smartflat/features/symbolic_barycenter/generalization/headroom.py` · **Tests:** `tests/test_headroom.py` (17; full suite **293 passed**)
- **Artifacts (git-ignored):** `$DATA_ROOT/outputs/symbolic_barycenter/generalization/frequency_controlled/{screen,calibration,order_null,order_null_secondary,incremental,negative_control_50salads,verdict}.csv`
- **Env:** conda `smartflat_repro`; `SMARTFLAT_DATA_ROOT=/home/perochon/data-gold-final NUMBA_THREADING_LAYER=workqueue MPLBACKEND=agg`

---

## 27.1 Motivation — why the prior nulls were not all comparable

The §15 ΔAUC order-null reports `delta_auc = AUC_intact − mean(AUC_shuffled)`, where the
shuffle preserves each sequence's symbol multiset. So it only asks "does order add anything
**beyond frequency**?" — a question that is only answerable when frequency leaves room.

Notebook `06m` ran that null on Breakfast's top-6 activities and got `order_helps` **0/60** —
but with `auc_intact = 1.0` on *all 60 cells*. The activities have near-disjoint action
vocabularies, so a bare unigram histogram already separates them perfectly, `delta_auc` is 0
**by arithmetic**, and the null could not have fired even if order mattered. **That 0/60 is
vacuous, not evidence** — and the same doubt hung over SDS2's 0/15.

## 27.2 The instrument — `headroom_table` + a two-sided band

`headroom_table(X, labels, G, D_G)` reports, per class pair, the out-of-fold **unigram-histogram
AUC** — the exact frequency channel the shuffle holds fixed — on the *same* folds and recipe as
`order_information`, then assigns a band:

| band | rule | is a null informative? |
|---|---|---|
| `saturated` | `hist_auc ≥ 0.95` | **No** — frequency already separates the classes (the 06m failure) |
| `floor` | `hist_auc ≤ 0.55` | **No** — frequency separates nothing; a null can't tell "order doesn't help" from "classes exchangeable" (06m failure mirrored) |
| `sweet` | otherwise | **Yes** — distinguishable *and* not saturated |

Gating the null on `hist_auc` is legitimate: it is a deterministic function of shuffle-**invariant**
quantities (a sequence's histogram *is* its multiset; labels and splits are untouched), so it is a
conditioning variable, not a statistic — no selection bias. Pinned as two executable
shuffle-invariance tests in `tests/test_headroom.py`.

## 27.3 Screen results — **the headline deliverable**

Breakfast was **deduplicated to distinct executions first**: its 1712 "videos" are 52 subjects ×
10 activities × up to 5 *simultaneous camera views*, and cross-view sequences are **98.8%
byte-identical** after RLE (symbol-set Jaccard median 1.000) → only **503** independent
executions. Leaving the duplicates in leaks copies across the non-grouped CV folds.

| dataset | n / pairs | `hist_auc` | band | reading |
|---|---|---|---|---|
| **SDS2 G=28** — HEALTHY_vs_RIL | 61 | **0.852** | sweet | — |
| **SDS2 G=28** — RIL_vs_TBI | 96 | **0.696** | sweet | — |
| **SDS2 G=28** — CONTROL_vs_PATIENT | 120 | **0.764** | sweet | — |
| **Breakfast** — all 45 activity pairs | ~100 ea. | **1.000** (min=med=max) | saturated ×45 | — |
| **GTEA** | 7 cls × 4 vids | — | untestable | 4 < 5-fold minimum → 0 screenable pairs |
| **50Salads** trial-split | 50 | **0.459** | floor | negative control, not a test (§27.5) |

**Consequences:**
- **SDS2's 0/15 null was a fair test.** All three comparisons have genuine headroom (0.70–0.85),
  so its negative is real evidence that order adds nothing beyond frequency on that cohort.
- **Breakfast's 0/60 was vacuous.** Every one of the 45 activity pairs is saturated, even
  deduplicated, even the most vocabulary-sharing pair. (Honest denominator: the loader RLEs, so
  `transition_features == run_transition_features` exactly → the 2×2 grid is one test ×4; "0/60"
  is 15 distinct tests ×4.)
- **No *naturally* frequency-controlled real task exists in this corpus.** Both candidate ideas
  from the prior phase are empirically dead: finer 50Salads labels sit at the floor; within-activity
  subject discrimination is impossible (≈1 execution per subject-activity after dedup).

## 27.4 Constructed frequency-controlled tasks

Since no natural task has headroom, we construct one — `restrict_to_shared_vocabulary` drops each
pair's class-exclusive **marker** symbols, keeping only the actions **both** classes perform, and
remaps to a compact alphabet (relative order preserved → output is a genuine subsequence). The
honest question narrows to: *given only the shared actions, does their **order** separate the classes?*

| task (shared-vocab restriction) | n | G_c | `hist_auc` | band | median len |
|---|---|---|---|---|---|
| friedegg vs pancake | 92 | 5 | 0.583 | **sweet** | 2 |
| friedegg vs scrambledegg | 99 | 7 | 0.634 | **sweet** | 4 |
| coffee vs milk | 36 | 2 | 0.636 | **sweet** | 2 |
| pancake vs scrambledegg | 90 | 6 | 0.491 | floor | 3 |

## 27.5 Guards — run *before* the real null

- **Calibration decoy** (`pooled_markov_surrogate`): keeps each sequence's multiset bit-identical
  (frequency–label link preserved) but resamples order from a **class-pooled** Markov model, so
  order carries *provably* no label information. Running the *unmodified* null on it measures the
  probe's empirical type-I rate at this dataset's operating point (10 decoys × 200 shuffles/task).
  **Result: type-I = 0/10 on all three primary tasks.** The probe is *not* anti-conservative here →
  positives are interpretable.
- **Real-data negative control** — 50Salads trial-1 vs trial-2 (same 25 subjects, same recipe
  twice; vocabulary shared by construction, exchangeable by design): **`order_helps` 0/4**, `auc_intact
  = 0.349`, `delta_auc = −0.146`. The probe correctly stays silent where nothing should distinguish
  the classes.

## 27.6 Order-null results

**Primary family** — one pre-registered combo (`run_transition × runlength`, the cleanest
pure-sequencing probe), 1000 shuffles, BH over the 3 sweet-band tasks (m=3 so BH is *declarable*:
at 200 shuffles the smallest attainable p, 1/201, exceeds BH's rank-1 threshold for m≥11):

| task | `hist_auc` | `auc_intact` | `delta_auc` | `p_perm` | `p_bh` | reject | beats freq |
|---|---|---|---|---|---|---|---|
| **friedegg vs pancake** | 0.583 | **0.913** | **+0.362** | 0.0010 | **0.0030** | ✅ | ✅ |
| **friedegg vs scrambledegg** | 0.634 | 0.704 | +0.145 | 0.0170 | **0.0255** | ✅ | ✅ |
| coffee vs milk | 0.636 | 0.578 | +0.032 | 0.411 | 0.411 | ❌ | ❌ |

**Secondary/exploratory family** — full 2×2 grid @ 200 shuffles on all constructed tasks,
BH within itself (**12/16 reject**). Consistent with the primary: friedegg-vs-pancake and
friedegg-vs-scrambledegg reject on all 4 combos; coffee-vs-milk on none. The **floor-band**
`pancake vs scrambledegg` also rejects on all 4 (`auc_intact ≈ 0.71` vs `hist_auc 0.49`) — a
positive that the two-sided band **pre-registered out of the primary family** (conservative: it
would have *added* a positive, not removed one).

**Mechanistic decode** (descriptive): friedegg oils the pan then cracks the egg
(`pour_oil → crack_egg`); pancake cracks the egg first (`crack_egg → pour_oil`). Same multiset,
opposite order — a signal the L1-normalized histogram literally cannot see.

## 27.7 Confirmatory probe — shuffle-free

`evaluate_incremental_ordering` trains on `hist` vs `both = [hist ⊕ transition]`, paired on identical
folds; because `both` **nests** `hist`, it is structurally immune to the blur artifact of §27.9.

| task | `mean_auc_hist` | `mean_auc_both` | `mean_delta` | 95% CI | wilcoxon_p |
|---|---|---|---|---|---|
| friedegg vs pancake | 0.583 | 0.901 | **+0.318** | [+0.219, +0.405] | 0.0003 |
| friedegg vs scrambledegg | 0.634 | 0.694 | +0.060 | [+0.010, +0.110] | 0.055 |
| coffee vs milk | 0.636 | 0.636 | 0.000 | [0, 0] | 1.000 |

Caveat: its bootstrap/Wilcoxon treat CV folds as independent (they are not), so its intervals are
anti-conservative in the *other* direction — which is why the decision rule requires **both** probes
to agree and reads effect sizes over p-values.

## 27.8 Verdict

Decision rule (pre-registered, all three required, sweet band only): `BH(p_perm) < 0.05` **AND**
`auc_intact > hist_auc` **AND** incremental `delta_ci_low > 0`.

| task | band | `hist_auc` | `auc_intact` | `p_bh` | incr. ΔCI-low | type-I | **ORDER HELPS** |
|---|---|---|---|---|---|---|---|
| **friedegg vs pancake** | sweet | 0.583 | 0.913 | 0.003 | +0.219 | 0.0 | ✅ **yes (decisive)** |
| **friedegg vs scrambledegg** | sweet | 0.634 | 0.704 | 0.026 | +0.010 | 0.0 | ✅ **yes (marginal)** |
| coffee vs milk | sweet | 0.636 | 0.578 | 0.411 | 0.000 | 0.0 | ❌ no |
| pancake vs scrambledegg | floor | 0.491 | — | — | — | — | (exploratory positive; out of primary) |

**On real (if constructed) data, order can help — decisively for friedegg-vs-pancake, marginally for
friedegg-vs-scrambledegg — with all guards clean.** The sweet-band negative (coffee-vs-milk) shows the
method does not fire indiscriminately.

**Scope (honest).** The restricted sequences are **2–4 symbols** long. The supportable claim is narrow:
*the relative order of a handful of shared actions distinguishes two recipes in a constructed contrast.*
It is **not** "order matters in procedural activity," and it does **not** overturn SDS2's negative — which
the screen has now shown was a fair test all along. Robustness banked: compact G vs full-alphabet give
identical `auc_intact`; result holds on deduplicated executions.

## 27.9 Methodological caveat — the shuffle null is anti-conservative when frequency signal is present

The shuffle tests "order is **uniform** given the multiset," which is strictly stronger than what we
want, "order is **independent of the label** given the multiset." Real sequences violate the former
massively (they are structured) while possibly satisfying the latter — and then shuffling **blurs** the
frequency signal (a sharp transition matrix degrades into a noisy one) and `delta_auc` goes positive with
*zero* order-label association. This is why every positive here is gated by the calibration decoy +
`auc_intact > hist_auc` + the nesting incremental probe. **It only threatens positives:** every existing
null (SDS2 0/15, Breakfast 0/60) is, if anything, *strengthened*. (Noted here; not yet written up as a
first-class paper finding.)

## 27.10 Reproduction

```bash
SMARTFLAT_DATA_ROOT=/home/perochon/data-gold-final NUMBA_THREADING_LAYER=workqueue MPLBACKEND=agg \
  jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=smartflat_repro \
  notebooks/06n_frequency_controlled_order.ipynb
# ~2.5 h on pomme; writes the 7 CSVs above; re-asserts 06m's committed artifacts are byte-identical.
```

Paper hooks: cite the **screen** for external validity (it licenses "SDS2's 0/15 means something" and
retires Breakfast 0/60 as evidence) — `PAPER_TODO §2.1` (Breakfast) and `§1.6` (negative ordering test as
a rigor signal).

---

## Poster-session summary

> ### Does temporal *order* ever beat symbol *frequency*? A headroom-gated test on real cooking data
>
> **Problem.** The order-vs-frequency null only means something when frequency doesn't already
> saturate the task. Prior "0/60 — order never helps" on Breakfast was run at a **frequency ceiling
> (AUC = 1.0)**: no room for the test to fire. Vacuous, not evidence.
>
> **Method.** A **frequency-headroom screen** — per class pair, measure the unigram-histogram AUC (the
> channel the null holds fixed) and band it: `saturated` / `floor` / `sweet`. Only run the order-null
> where the screen shows headroom. Guard every positive with a class-pooled **calibration decoy**, a
> real-data **negative control**, and a shuffle-free confirmatory probe.
>
> **What we found**
> - 🟢 **The screen re-reads two old nulls.** SDS2 (hist-AUC 0.70–0.85) **had headroom → its 0/15 was a
>   fair test.** Breakfast (**45/45 pairs at hist-AUC = 1.000**, even deduped 1712→503 executions) **never
>   did → 0/60 was vacuous.** GTEA untestable; 50Salads at the floor.
> - 🟢 **No natural frequency-controlled real task exists here** — the headline negative result.
> - 🔵 **On a constructed contrast, order DOES help.** Restrict to shared actions: **friedegg-vs-pancake
>   hist 0.58 → AUC 0.91 (Δ +0.36, p=0.003)**, friedegg-vs-scrambledegg marginally. Mechanism:
>   `pour_oil→crack_egg` vs `crack_egg→pour_oil` — invisible to a histogram.
> - ✅ **Guards clean:** decoy type-I = 0/30, negative control 0/4, both probes agree. Sweet-band
>   coffee-vs-milk is **negative** → not firing indiscriminately.
>
> **Take-home.** *Order rarely helps because real activity classes are usually separable by frequency
> alone — but where you engineer frequency away, a genuine, mechanistically interpretable ordering signal
> survives.* Scope: the constructed sequences are 2–4 symbols; this refines, not overturns, the negative.
>
> **Bonus (methods).** The frequency-preserving shuffle null is **anti-conservative when frequency signal
> is present** (it blurs frequency, not just order) — so it needs a calibration decoy before any positive.
> This *strengthens* every previously reported null.
