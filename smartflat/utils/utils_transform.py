import numpy as np
from sklearn.preprocessing import StandardScaler

def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score a multivariate time-series (n_samples, n_features)"""
    zscore_tool = StandardScaler()
    x = zscore_tool.fit_transform(x)
    return x