# Barycenter methods — how each treats the discrete (categorical) symbol space

**Purpose.** A paper-writing reference for the barycenter methods-comparison (RESULTS_HANDOFF §17/§18,
notebooks `06g`/`06i`). It answers one fairness question precisely: *are all the benchmarked averagers
developed with full respect to the discrete nature of our symbolic state sequences, and how is that handled?*
Verified against the installed implementations in
`smartflat/features/symbolic_barycenter/baselines.py` (2026-07-02, branch `barycenter-quality-fgw`).

**One-line answer.** No — and by design. The benchmarked averagers split into **two families**: one that
stays in the discrete symbol space throughout (native-categorical), and one that optimises in a **continuous
relaxation** and imposes discreteness only by a final nearest-neighbour **decode**. The comparison is still
fair *as a benchmark* (common yardstick + each method on its own native distance), but the *mechanism* differs
and the paper should say so, because it is exactly what several of the §17/§18 numbers reflect.

---

## Family A — native-categorical (never leave symbol space)

Each of these operates on integer symbols directly (or on symbol-derived objects), and where it emits a
sequence, every output position is a **genuine symbol** chosen by voting / median / medoid. The reference is
a valid symbol string at every step.

| method (registry key) | builder | how discreteness is handled | output object | re-discretised |
|---|---|---|---|---|
| **`tw_twe_mode`** (the paper method) | `barycenter_mode_dba` | rTWE-align integer sequences to an integer reference; per-position **hard majority vote** of aligned symbols; reference init = within-group rTWE medoid | symbol sequence | **every iteration** |
| `shape_dba` | `barycenter_soft_mode_dba` | as above but a **soft** vote `softmax(-β·D_G[:,s])` accumulated per position, then **argmax** over candidate symbols (β→∞ recovers the hard mode) | symbol sequence | **every iteration** |
| `majority_voting` | `barycenter_majority_voting` | lock-step per-position mode (no alignment) | symbol sequence | n/a (direct) |
| `edit_median` | `barycenter_edit_median` | Levenshtein median string via local search; substitutions drawn from the **actual alphabet** | symbol sequence | n/a (stays symbolic) |
| `k_medoid` | `barycenter_k_medoid` | returns an **actual observed member** sequence (argmin of within-group pairwise rTWE) | symbol sequence (a real data point) | n/a (is a real sequence) |
| `wasserstein` | `barycenter_wasserstein` | entropic-OT barycenter of **symbol-frequency histograms** (ground cost `D_G`) | histogram over symbols (**no order**) | n/a |
| `transition` | `barycenter_transition_matrix` | row-normalised **bigram transition matrix** over symbols | matrix (**no sequence**) | n/a |
| **`msa_consensus`** (F·S3) | `barycenter_msa_consensus` | **center-star MSA** (Gusfield 1993): rTWE-align every member to the medoid, merge with insertion columns ("once a gap, always a gap"); **profile per-column consensus** — Laplace-pseudocount argmax (Durbin et al. 1998) — kept only where non-gap **occupancy ≥ 50%** (match-state rule) | symbol sequence | n/a (voting **is** the discrete step) |

Notes:
- `tw_twe_mode` is the method described by the accumulator + majority-vote-under-TW-TWE design. The current
  repo **reimplements it natively** via the vendored `rtwe_alignment_path` — it does **not** use the forked
  aeon `elastic_barycenter_average(method='petitjean', distance='twe', precomputed_distances=D_G)` the thesis
  used. That fork is only *referenced* (and mean-reconstructed) in the diagnostic `barycenter_mean_rtwe_dba`,
  which is **not** in the §17/§18 comparison table.
- `wasserstein` and `transition` are categorical-respecting but are **different objects** (a distribution and
  a matrix): they carry no per-position symbol sequence, so the harness reports NaN for the sequence-only
  axes (`inertia_rtwe`, `n_segments`, etc.).

---

## Family B — continuous relaxation + decode (the `D_G`-row embedding)

Each symbol `s` is embedded as its **`D_G` row** `D_G[s]` (a vector of distances to all symbols) —
`embed_symbolic_to_real`. A real-valued barycenter is optimised over these vectors, then decoded back to hard
symbols by **nearest `D_G` row** (`project_real_to_symbolic`, Euclidean) — or, for FGW, nearest prototype in
the feature space.

| method (registry key) | builder | continuous averaging step | re-discretised |
|---|---|---|---|
| `dba_dtw` | `barycenter_dba_dtw` | DBA (Petitjean): **arithmetic mean of aligned real vectors** per position | **only at the end** |
| `soft_dtw` (hand-rolled) | `barycenter_soft_dtw` | soft-DTW Fréchet mean by gradient descent | **only at the end** |
| **`soft_dtw_bary`** (tslearn) | `barycenter_softdtw` | soft-DTW barycenter (L-BFGS-B; Cuturi & Blondel 2017) | **only at the end** |
| **`ssg`** (tslearn) | `barycenter_ssg` | stochastic-subgradient DTW average (Schultz & Jain 2018) | **only at the end** |
| `fgw_mds` / `fgw_onehot` | `barycenter_fgw` | FGW feature centroid; nodes = MDS(`D_G`) or one-hot features | decode to nearest prototype **at the end** |

### Why this is not "discrete-respecting" during optimisation
1. **The mean is between symbols.** The averaged vector at a position is a convex combination of symbol
   embeddings that generally corresponds to **no actual symbol** — the classic *"mean of category 2 and
   category 70 → category 36"* pathology, which the `barycenter_mean_rtwe_dba` docstring itself names as the
   motivation for mode-DBA. Discreteness is imposed only by the final snap.
2. **DBA-family drifts; mode-DBA does not.** `dba_dtw` / `soft_dtw` / `soft_dtw_bary` / `ssg` accumulate
   continuous drift across all iterations and snap **once** at the end; `tw_twe_mode` / `shape_dba` re-project
   to a real symbol **every** iteration. Algorithmic difference, not cosmetic.
3. **The decode is Euclidean in distance-profile space, not ground-cost-optimal.**
   `project_real_to_symbolic` picks the symbol whose *`D_G`-row* is Euclidean-nearest to the mean profile. The
   mean of distance-profiles need not be a valid profile of any symbol, so the snap is a proxy — not the
   `D_G`/Wasserstein-optimal decode. (`fgw_onehot` sidesteps `D_G` in its feature term → its decode is
   nearest one-hot = argmax of the soft assignment; `fgw_mds` decodes in MDS space.)

---

## Is the benchmark still fair?

**Yes, with an explicit caveat.** The harness (`barycenter_quality.score_barycenter_quality`) keeps it fair
two ways:
- a **common yardstick** — rTWE within-group inertia computed on the **decoded symbol sequences** — so every
  sequence-emitting method is scored in the same warped geometry; and
- **each method on its own native distance** (`inertia_native`), reported alongside — the §17/§18 framing of
  "an honest tradeoff, not a single winner."

What to add in the paper is the **mechanism split** above, because the metrics reflect it: the
continuous-embed methods lose **frequency fidelity** (`soft_dtw_bary` 0.165, `ssg` 0.138 vs the histogram's
0.027) and **entropy** partly *because* the smoothed continuous mean snaps many positions to a few centroid
symbols.

**Do not over-claim.** Mode-collapse is **not** unique to Family B — the hard-voting `tw_twe_mode` is the most
collapsed method in the table (13 distinct symbols, 3.02 bits). The two families collapse by **different
mechanisms**: dominant-symbol voting (Family A hard-mode) vs snap-of-a-smoothed-mean (Family B). The quality
table captures both; the honest statement is a *mechanism* difference, not a "Family B is worse" claim.

### Where it shows in the §17/§18 numbers (G=28, L=128, gated preview)

| method | family | freq_fidelity ↓ | entropy (bits) | rTWE inertia ↓ | n_distinct | n_segments |
|---|---|---|---|---|---|---|
| wasserstein (histogram) | A (freq only) | **0.027** | 3.73 | — | 28 | — |
| fgw_onehot | B | 0.037 | 3.55 | 75.8 | 18 | 111 |
| fgw_mds | B | 0.044 | 3.50 | 77.8 | 17 | 113 |
| ssg | B | 0.138 | 3.24 | 47.1 | 16 | 65 |
| soft_dtw_bary | B | 0.165 | 3.17 | 48.7 | 14 | 26 |
| tw_twe_mode (paper) | A | 0.120 | 3.02 | **44.8** | 13 | 57 |

(group-sequence entropy reference ≈ 3.78 bits.) Read: Family A's alignment-voting methods and Family B's
DTW-family methods are both **rTWE-compact** (≈ 45–49) but both **less frequency-faithful / lower-entropy**
than the frequency histogram and the anti-collapse FGW — the compactness-vs-fidelity frontier, reached from
two different directions (discrete voting vs continuous-mean-then-snap).

---

## Levers to tighten fairness (planned; see the kickoff)

Roughly in increasing effort. **Lever 1 is done** (this document + the §18.4 note + the 06i family column).
Levers 2 and 3 are deferred to a fresh session — `.claude/prompts/kickoff-K-discreteness-fairness.md`.

1. **Document the mechanism split** (done). No code risk.
2. **Ground-cost-consistent decode.** Replace the Euclidean `project_real_to_symbolic` with a `D_G`-argmin
   decode for the embedding methods, so Family B's snap uses the same geometry the pipeline scores in. Report
   as a decode-ablation (Euclidean vs `D_G`) so the change is auditable, not silent.
3. **Per-iteration re-discretisation** for `dba_dtw` / `ssg` (snap after each update, not only at the end),
   giving them the same "stay categorical throughout" property as mode-DBA — an apples-to-apples
   *categorical variant* reported **alongside** the standard continuous ones (not replacing them).

**Guardrails for 2 & 3:** additive (new variants beside the existing methods, existing keys unchanged); no
p-hacking (report the ablation both ways, per-axis); reuse the §17 harness; tests-first; E/F/G frozen suites
must stay green.
