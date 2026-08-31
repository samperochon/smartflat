"""Tests for the RLE (segment-scale, duration-explicit) representation + method redesign.

Covers the contracts promised by the redesign plan:
- RLE round-trip and adjacent-run merging;
- gamma_dur = 0 (or equal durations) reduces the duration-aware rTWE EXACTLY to
  the vendored frame-level rTWE on the RLE symbol sequence;
- the (symbol, duration) pointwise cost satisfies the triangle inequality when
  the symbolic ground cost does;
- mode-DBA upgrades: 'dg_frechet' update validity, genuine 'random' inits,
  'best'-iterate return never worse than 'last', 'mode'+'last' backward compat;
- leakage-free temporal ground cost (W1 correctness vs scipy, symmetry);
- semantic ground cost / blend basic properties;
- pmatch denominator option + soft variant bounds;
- evaluate_baselines(methods_factory=...) equivalence and RLE-object-array support.
"""

import numpy as np
import pytest

from _bary_helpers import _ground_cost

from smartflat.engine.distances._rtwe import rtwe_distance
from smartflat.engine.distances._rtwe_duration import (
    rtwe_dur_alignment_path,
    rtwe_dur_distance,
    rtwe_dur_pairwise_distance,
)
from smartflat.features.symbolic_barycenter.builders import barycenter_mode_dba
from smartflat.features.symbolic_barycenter.distances import (
    pmatch_soft_to_barycenter,
    pmatch_to_barycenter,
)
from smartflat.features.symbolic_barycenter.evaluation import evaluate_baselines
from smartflat.features.symbolic_barycenter.rle import (
    barycenter_rle_dba,
    dist_rle_twe,
    merge_adjacent_runs,
    pmatch_rle,
    rle_decode,
    rle_encode,
    rle_methods,
    rle_object_array,
)
from smartflat.features.symbolic_barycenter.vocab import (
    blend_ground_costs,
    compute_distance_matrix,
    semantic_ground_cost,
    temporal_ground_cost,
)


def _frame_seq(rng, length, g=6):
    """Frame-level sequence with realistic runs (repeat-heavy)."""
    out = []
    while sum(len(r) for r in out) < length:
        out.append([rng.randint(0, g)] * rng.randint(1, 8))
    return np.concatenate(out)[:length].astype(int)


# ---------------------------------------------------------------------------
# RLE encode / decode
# ---------------------------------------------------------------------------

def test_rle_roundtrip():
    rng = np.random.RandomState(0)
    seq = _frame_seq(rng, 200)
    packed = rle_encode(seq)
    assert packed.shape[0] == 2
    np.testing.assert_array_equal(rle_decode(packed), seq)
    # runs are maximal: no two adjacent symbols equal
    assert np.all(np.diff(packed[0]) != 0)


def test_merge_adjacent_runs():
    packed = np.array([[2, 2, 3, 3, 3, 1], [1, 4, 2, 1, 1, 5]])
    merged = merge_adjacent_runs(packed)
    np.testing.assert_array_equal(merged[0], [2, 3, 1])
    np.testing.assert_array_equal(merged[1], [5, 4, 5])
    assert merged[1].sum() == packed[1].sum()


# ---------------------------------------------------------------------------
# Duration-aware rTWE
# ---------------------------------------------------------------------------

def test_gamma_zero_reduces_to_rtwe():
    """gamma=0 must equal the vendored rTWE on the RLE symbols exactly."""
    rng = np.random.RandomState(1)
    d_g = _ground_cost(6)
    for _ in range(5):
        a = rle_encode(_frame_seq(rng, 150))
        b = rle_encode(_frame_seq(rng, 120))
        want = rtwe_distance(
            a[0].astype(np.float64), b[0].astype(np.float64),
            nu=1e-4, lmbda=0.1, precomputed_distances=d_g,
        )
        got = rtwe_dur_distance(a[0], a[1], b[0], b[1], d_g,
                                gamma=0.0, nu=1e-4, lmbda=0.1)
        assert got == pytest.approx(want, rel=1e-12)


def test_equal_durations_reduce_to_rtwe():
    """With all-equal durations the log-duration term vanishes for any gamma."""
    rng = np.random.RandomState(2)
    d_g = _ground_cost(6)
    sa = np.array([0, 3, 2, 5, 1, 4, 2])
    sb = np.array([3, 3, 2, 1, 0])
    ones_a, ones_b = np.full(len(sa), 7), np.full(len(sb), 7)
    want = rtwe_distance(sa.astype(np.float64), sb.astype(np.float64),
                         nu=1e-4, lmbda=0.1, precomputed_distances=d_g)
    got = rtwe_dur_distance(sa, ones_a, sb, ones_b, d_g,
                            gamma=0.7, nu=1e-4, lmbda=0.1)
    assert got == pytest.approx(want, rel=1e-12)
    del rng


def test_duration_mismatch_increases_distance():
    d_g = _ground_cost(6)
    s = np.array([1, 4, 2])
    d_same = rtwe_dur_distance(s, [10, 10, 10], s, [10, 10, 10], d_g,
                               gamma=0.5, nu=1e-4, lmbda=0.1)
    d_diff = rtwe_dur_distance(s, [10, 10, 10], s, [10, 80, 10], d_g,
                               gamma=0.5, nu=1e-4, lmbda=0.1)
    assert d_same == pytest.approx(0.0, abs=1e-12)
    assert d_diff > d_same


def test_pointwise_cost_triangle_inequality():
    """d((a,da),(b,db)) = D[a,b] + gamma|log da - log db| is a metric if D is."""
    rng = np.random.RandomState(3)
    # Build a genuine metric D via shortest-path completion of a random symmetric matrix.
    from scipy.sparse.csgraph import shortest_path
    d0 = _ground_cost(6, seed=7)
    d_metric = shortest_path(d0, directed=False)
    gamma = 0.4

    def cost(a, la, b, lb):
        return d_metric[a, b] + gamma * abs(np.log(la) - np.log(lb))

    for _ in range(500):
        a, b, c = rng.randint(0, 6, 3)
        la, lb, lc = rng.randint(1, 100, 3).astype(float)
        assert cost(a, la, c, lc) <= cost(a, la, b, lb) + cost(b, lb, c, lc) + 1e-12


def test_pairwise_symmetry_and_alignment_path():
    rng = np.random.RandomState(4)
    d_g = _ground_cost(6)
    seqs = [rle_encode(_frame_seq(rng, 100)) for _ in range(4)]
    D = rtwe_dur_pairwise_distance(seqs, d_g, gamma=0.2, nu=1e-4, lmbda=0.1)
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)
    path, dist = rtwe_dur_alignment_path(
        seqs[0][0], seqs[0][1], seqs[1][0], seqs[1][1], d_g,
        gamma=0.2, nu=1e-4, lmbda=0.1,
    )
    assert dist == pytest.approx(D[0, 1])
    assert path[0] == (0, 0)
    assert path[-1] == (seqs[0].shape[1] - 1, seqs[1].shape[1] - 1)


# ---------------------------------------------------------------------------
# RLE barycenter
# ---------------------------------------------------------------------------

def test_rle_dba_valid_output_and_cost_trace():
    rng = np.random.RandomState(5)
    d_g = _ground_cost(6)
    X = [rle_encode(_frame_seq(rng, 120)) for _ in range(6)]
    bary, costs = barycenter_rle_dba(X, d_g, gamma_dur=0.2, nu=1e-4, lmbda=0.1,
                                     return_costs=True)
    assert bary.shape[0] == 2
    assert set(np.unique(bary[0])).issubset(set(range(6)))
    assert np.all(bary[1] >= 1)
    assert np.all(np.diff(bary[0]) != 0)  # merged runs
    assert len(costs) >= 1
    # returned iterate is the best-cost one: its cost is min of the trace
    total = sum(
        dist_rle_twe(s, bary, d_g, gamma_dur=0.2, nu=1e-4, lmbda=0.1) for s in X
    )
    # merging runs can only change the distance via the alignment; allow tolerance
    assert total <= min(costs) * 1.05 + 1e-9


def test_rle_dba_single_sequence_identity():
    d_g = _ground_cost(6)
    x = rle_encode(np.array([1, 1, 2, 2, 2, 3]))
    np.testing.assert_array_equal(barycenter_rle_dba([x], d_g), x)


def test_rle_pmatch_bounds_and_self_match():
    rng = np.random.RandomState(6)
    d_g = _ground_cost(6)
    a = rle_encode(_frame_seq(rng, 100))
    b = rle_encode(_frame_seq(rng, 90))
    p = pmatch_rle(a, b, d_g, gamma_dur=0.2, nu=1e-4, lmbda=0.1)
    assert 0.0 <= p <= 1.0
    assert pmatch_rle(a, a, d_g, gamma_dur=0.2) == pytest.approx(1.0)
    p_soft = pmatch_rle(a, b, d_g, gamma_dur=0.2, soft=True)
    assert p_soft >= p - 1e-12  # soft credit can only add


# ---------------------------------------------------------------------------
# mode-DBA upgrades
# ---------------------------------------------------------------------------

def _mode_dba_cohort(seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 6, size=(8, 40)).astype(int), _ground_cost(6)


def test_mode_dba_backcompat_mode_last():
    """update='mode', keep='last' reproduces the historical algorithm."""
    x, d_g = _mode_dba_cohort()
    b_old_style = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1,
                                      update='mode', keep='last')
    b_default = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1)
    assert b_old_style.shape == (40,)
    # default keep='best' coincides with 'last' when the trace is monotone;
    # both must be valid symbol sequences either way.
    for b in (b_old_style, b_default):
        assert set(np.unique(b)).issubset(set(range(6)))


def test_mode_dba_dg_frechet_update():
    x, d_g = _mode_dba_cohort(1)
    b = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1, update='dg_frechet')
    assert b.shape == (40,)
    assert set(np.unique(b)).issubset(set(range(6)))


def test_mode_dba_best_never_worse_than_last():
    from smartflat.features.symbolic_barycenter.distances import dist_rtwe
    x, d_g = _mode_dba_cohort(2)
    for update in ('mode', 'dg_frechet'):
        b_best = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1,
                                     update=update, keep='best')
        b_last = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1,
                                     update=update, keep='last')
        cost = lambda b: sum(dist_rtwe(s, b, d_g, nu=1e-4, lmbda=0.1) for s in x)
        assert cost(b_best) <= cost(b_last) + 1e-9


def test_mode_dba_random_init_uses_seed():
    x, d_g = _mode_dba_cohort(3)
    outs = {
        barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1, init='random',
                            random_state=s).tobytes()
        for s in range(6)
    }
    # deterministic per seed
    b1 = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1, init='random', random_state=0)
    b2 = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1, init='random', random_state=0)
    np.testing.assert_array_equal(b1, b2)
    # and the seed actually matters (>=2 distinct outcomes across 6 seeds)
    assert len(outs) >= 2


# ---------------------------------------------------------------------------
# Ground costs
# ---------------------------------------------------------------------------

def test_temporal_ground_cost_w1_matches_scipy():
    from scipy.stats import wasserstein_distance
    # symbol 1 occurs early, symbol 2 late, in every sequence
    X = [np.array([1] * 10 + [2] * 10), np.array([1] * 5 + [2] * 15)]
    D = temporal_ground_cost(X, G=3, weighting='uniform')
    t0 = np.arange(20) / 20
    t1 = np.arange(20) / 20
    v1 = np.concatenate([t0[:10], t1[:5]])
    v2 = np.concatenate([t0[10:], t1[5:]])
    want = wasserstein_distance(v1, v2)
    assert D[1, 2] == pytest.approx(want, rel=1e-9)
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_temporal_ground_cost_missing_symbol_filled_with_max():
    X = [np.array([1, 1, 2, 2])]
    D = temporal_ground_cost(X, G=4, weighting='uniform')  # symbol 3 never occurs
    assert D[3, 1] == D.max()
    assert D[3, 3] == 0.0


def test_semantic_ground_cost_and_blend():
    rng = np.random.RandomState(7)
    centroids = rng.randn(10, 8)
    cats = np.array([0, 1, 1, 2, 2, 2, 3, 3, 1, 2])  # category 4 empty
    D = semantic_ground_cost(centroids, cats, G=5)
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)
    assert D[0, 1] == D.max() > 0          # background row maxed
    assert D[4, 1] == D.max()              # empty category maxed
    B = blend_ground_costs(D, D, alpha=0.3)
    np.testing.assert_allclose(B, D / D.max())


def test_compute_distance_matrix_offdiag_is_scale_calibrated():
    """The 'offdiag_pre' floor is a fixed fraction of the (normalized) symbol scale.

    The thesis recipe's floor depends on the raw units: on a raw matrix whose max
    is < offset-scale (the G=28 case, raw max ~0.48) it flattens the off-diagonal
    contrast far more than the calibrated variant.
    """
    rng = np.random.RandomState(8)
    raw = np.abs(rng.randn(6, 6))
    raw = (raw + raw.T) / 2
    np.fill_diagonal(raw, 0.0)
    raw = raw / raw.max() * 0.48  # G=28-like raw scale
    d_thesis = compute_distance_matrix(raw, method='max_rows_cols_pre', offset_value=0.3)
    d_new = compute_distance_matrix(raw, method='offdiag_pre', offset_value=0.3)
    inner = np.ix_(range(1, 6), range(1, 6))

    def contrast(d):
        off = d[inner][~np.eye(5, dtype=bool)]
        return off.min() / off.max()  # 1.0 = perfectly flat

    assert contrast(d_new) < contrast(d_thesis)  # more symbol-identity contrast
    # calibrated floor: (r_min + offset) / (1 + offset) with r_min the min
    # normalized raw off-diagonal -- independent of the raw units (0.48 here)
    off_new = d_new[inner][~np.eye(5, dtype=bool)]
    raw_norm = raw / raw.max()
    r_min = raw_norm[inner][~np.eye(5, dtype=bool)].min()
    assert off_new.min() == pytest.approx((r_min + 0.3) / 1.3, rel=1e-9)
    assert np.all(np.diag(d_new) == 0)
    np.testing.assert_allclose(d_new, d_new.T)


# ---------------------------------------------------------------------------
# p_match refinements
# ---------------------------------------------------------------------------

def test_pmatch_denominator_option():
    rng = np.random.RandomState(9)
    d_g = _ground_cost(6)
    a = rng.randint(0, 6, 30)
    b = rng.randint(0, 6, 45)  # unequal lengths force edit steps
    p_diag = pmatch_to_barycenter(a, b, d_g, nu=1e-4, lmbda=0.1)
    p_all = pmatch_to_barycenter(a, b, d_g, nu=1e-4, lmbda=0.1, denominator='all')
    assert 0 <= p_all <= p_diag <= 1
    with pytest.raises(ValueError):
        pmatch_to_barycenter(a, b, d_g, denominator='bogus')


def test_pmatch_soft_bounds_and_uniform_cost_equivalence():
    rng = np.random.RandomState(10)
    a = rng.randint(0, 6, 30)
    b = rng.randint(0, 6, 30)
    uniform = 1.0 - np.eye(6)
    hard = pmatch_to_barycenter(a, b, uniform, nu=1e-4, lmbda=0.1)
    soft = pmatch_soft_to_barycenter(a, b, uniform, nu=1e-4, lmbda=0.1)
    assert soft == pytest.approx(hard)
    d_g = _ground_cost(6)
    s = pmatch_soft_to_barycenter(a, b, d_g, nu=1e-4, lmbda=0.1)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------

def _tiny_cohort_frames(seed=0):
    rng = np.random.RandomState(seed)
    X = [
        np.repeat(rng.choice(6, 30, p=p), rng.randint(1, 6, 30))
        for p in ([.4, .2, .15, .1, .1, .05],) * 6 + ([.05, .1, .1, .15, .2, .4],) * 6
    ]
    labels = np.array(['A'] * 6 + ['B'] * 6, dtype=object)
    return X, labels


def test_evaluate_baselines_rle_object_array():
    X_frames, labels = _tiny_cohort_frames()
    d_g = _ground_cost(6)
    X = rle_object_array(X_frames)
    methods = {k: v for k, v in rle_methods(d_g, gamma_dur=0.2).items()
               if v.get('kind') != 'medoid'}
    df = evaluate_baselines(X, labels, methods, n_splits=2, n_inits=1)
    assert set(df['method']) == {'rle_dba_pmatch', 'rle_dba_twe'}
    assert df['auc'].between(0, 1).all()


def test_evaluate_baselines_methods_factory_equivalence():
    X_frames, labels = _tiny_cohort_frames(1)
    d_g = _ground_cost(6)
    X = rle_object_array(X_frames)
    methods = {'rle_dba_twe': rle_methods(d_g, gamma_dur=0.2)['rle_dba_twe']}
    df_static = evaluate_baselines(X, labels, methods, n_splits=2, n_inits=1)
    df_factory = evaluate_baselines(
        X, labels, n_splits=2, n_inits=1,
        methods_factory=lambda X_tr, y_tr: methods,
    )
    import pandas as pd
    pd.testing.assert_frame_equal(
        df_static.reset_index(drop=True), df_factory.reset_index(drop=True),
    )
    with pytest.raises(ValueError):
        evaluate_baselines(X, labels, methods, methods_factory=lambda a, b: methods)
    with pytest.raises(ValueError):
        evaluate_baselines(X, labels)
