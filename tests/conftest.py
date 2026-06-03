"""Shared fixtures for distance module tests."""

import numpy as np
import pytest


@pytest.fixture
def D5():
    """5x5 symmetric distance matrix with zero diagonal (prototypes 0-4)."""
    rng = np.random.RandomState(42)
    D = rng.rand(5, 5)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D
