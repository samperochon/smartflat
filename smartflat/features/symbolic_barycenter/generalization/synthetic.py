"""Controlled synthetic generalization datasets (PAPER_TODO §2.3 + RESULTS_HANDOFF §14).

Two fully-offline datasets, promoted and generalized from the inline fixtures in
``tests/test_order_evaluation.py`` (positive/negative controls):

- :func:`make_order_discriminative_dataset` — the **validity-proof** set. Two groups
  whose sequences share an **identical symbol multiset** (so per-sequence frequency is
  exactly matched — the token/run-length shuffle nulls are exact) and differ **only in
  arrangement** (blocked vs interleaved). Order is *provably* the only between-group
  signal, so ``order_information`` must report ΔAUC ``ci_low > 0`` — the clean converse
  of the SDS2 "0/15 order-null" (§15). A ``jitter`` knob dials the planted order from
  full (0.0) down to none (1.0), giving a signal-strength sweep.

- :func:`make_exact_recovery_dataset` — the **D1** set ("required for a tier-1 venue"):
  a *known* ground-truth centre plus members drawn at several edit-noise regimes, so an
  averager can be scored on how well it recovers the planted centre as noise rises. At
  zero noise every member equals the centre, so a correct averager recovers it exactly.

Both return the standard ``(X, labels, G, D_G)`` contract consumed by
:func:`..suite.run_generalization_suite`; ``D_G`` is a uniform (metric) ground cost
since synthetic symbols carry no intrinsic dissimilarity.
"""
import numpy as np


def uniform_ground_cost(G):
    """``(G, G)`` metric ground cost: 0 on the diagonal, 1 off it (unit discrete metric)."""
    D = np.ones((G, G), dtype=np.float64) - np.eye(G)
    return D


def make_order_discriminative_dataset(n_per_group=40, L=48, G=4, grammar='order',
                                      jitter=0.0, seed=0):
    """Two groups, identical per-sequence frequency, different transition grammar.

    Every sequence (both groups) is an arrangement of the **same multiset** — each of the
    ``G`` symbols repeated ``L // G`` times — so token/run-length shuffling is exact and
    frequency carries no between-group signal. Two constructions:

    - ``grammar='order'`` (default): group ``ascending`` orders the runs ``0,1,…,G-1`` and
      ``descending`` reverses them. Frequency **and** run-length dwell are identical, so the
      only between-group difference is **pure sequencing** — every order probe (both nulls,
      both features) detects it. This is the crisp converse of the SDS2 §15 order-null.
    - ``grammar='dwell'``: ``blocked`` (long runs) vs ``interleaved`` (unit runs). Frequency
      matched but the difference is in **run-length/dwell**, so only the dwell-sensitive
      probes (``transition``×``token``) fire — a useful contrast that localizes the signal.

    ``jitter`` ∈ [0, 1] is the probability that a sequence is emitted instead as a uniform
    random permutation of its multiset (a pure order-null draw), so ``jitter=0`` plants full
    order and ``jitter=1`` removes all between-group order signal (frequency stays matched).

    Returns ``(X, labels, G, D_G)`` (non-SDS2 labels → the harness yields a single pairwise
    comparison) with a uniform ``D_G``.
    """
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(f"jitter must be in [0, 1], got {jitter}")
    if L % G != 0:
        raise ValueError(f"L ({L}) must be a multiple of G ({G}) for an exact matched multiset")
    per = L // G
    if grammar == 'order':
        bases = (('ascending', np.repeat(np.arange(G), per)),       # runs 0,1,..,G-1
                 ('descending', np.repeat(np.arange(G)[::-1], per)))  # runs G-1,..,1,0
    elif grammar == 'dwell':
        bases = (('blocked', np.repeat(np.arange(G), per)),         # 0..0 1..1 (long runs)
                 ('interleaved', np.tile(np.arange(G), per)))       # 0 1 2 .. (unit runs)
    else:
        raise ValueError(f"grammar must be 'order' or 'dwell', got {grammar!r}")

    # 'order' rolls by whole runs (step = per) so every sequence keeps exactly G runs of
    # length `per` -> dwell is identical per sequence and the only signal is run order.
    roll_step = per if grammar == 'order' else 1
    n_rolls = L // roll_step
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for name, base in bases:
        for _ in range(n_per_group):
            if rng.random() < jitter:
                seq = rng.permutation(base)                              # order-null: random arrangement
            else:
                seq = np.roll(base, roll_step * int(rng.integers(0, n_rolls)))  # planted order
            X.append(seq.astype(int))
            labels.append(name)
    return X, np.array(labels, dtype=object), G, uniform_ground_cost(G)


def _random_center(G, L, rng, min_run=3, max_run=8):
    """A fixed 'prototypical execution': runs of random symbols (structured, order-rich)."""
    seq = []
    while len(seq) < L:
        sym = int(rng.integers(0, G))
        run = int(rng.integers(min_run, max_run + 1))
        seq.extend([sym] * run)
    return np.array(seq[:L], dtype=int)


def _corrupt(center, p_sub, rng):
    """Substitution edit-noise: each position swapped to a different symbol w.p. ``p_sub``."""
    G = int(center.max()) + 1
    out = center.copy()
    for i in range(len(out)):
        if rng.random() < p_sub:
            choices = [s for s in range(G) if s != out[i]]
            out[i] = int(rng.choice(choices)) if choices else out[i]
    return out


def make_exact_recovery_dataset(G=8, L=60, n_per=30,
                                noise_levels=(0.0, 0.1, 0.2, 0.35), seed=0):
    """D1: a known centre + members at several substitution-noise regimes.

    Returns ``(regimes, center, G, D_G)`` where ``regimes`` maps each noise level ``p``
    to ``(X, labels)`` — ``n_per`` members corrupted from the shared ground-truth
    ``center`` at rate ``p`` (all length ``L``; ``labels`` all ``'regime_{p}'``). Score
    an averager by the distance between its barycenter of ``X`` and ``center`` (exact at
    ``p = 0``, degrading with ``p``).
    """
    rng = np.random.default_rng(seed)
    center = _random_center(G, L, rng)
    regimes = {}
    for p in noise_levels:
        X = [_corrupt(center, p, rng) for _ in range(n_per)]
        labels = np.array([f'regime_{p}'] * n_per, dtype=object)
        regimes[p] = (X, labels)
    return regimes, center, G, uniform_ground_cost(G)
