"""Helper functions to transform input or target data (augmentations on time series, tranformation of the label, etc.)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def create_smooth_label_from_continuous_target(
    target_vals: pd.Series,
    min_value: float,
    max_value: float,
    bin_size: float,
    kernel_std: float,
) -> pd.Series:
    """this method takes a pandas series of a continuous target
    and returns a smooth binned label of the the target, representing
    the probability of each bin/class

    args:
        - min_value: min target value for the binning
        - max_value: max target value for the binning
        - bin_size: size of each bin (or each class)
        - kernel_std: std of gaussian distibution around the observed target

    returns
        - smooth_labels: pandas series of smooth labels
        containing N = (max_value - min_value) // bin_size classes,
        and each value represents the N-dim probability distribution
        of these classes.

    """
    # set bins
    bin_vals = np.arange(min_value, max_value, bin_size)

    # apply smoothing
    smooth_labels = target_vals.apply(
        lambda x: make_gaussian_prob_vec(
            bin_vals=bin_vals,
            loc=x,
            scale=kernel_std,
            do_normalize=True,
        )
    )

    return smooth_labels


#
# utils
#


def whiten_matrix(X, epsilon=1e-10):
    """
    Perform whitening on the input matrix X.

    Parameters:
    X (numpy.ndarray): 2D input matrix of shape (N_samples, N_dimensions)
    epsilon (float): Small constant to add to the singular values for numerical stability

    Returns:
    numpy.ndarray: Whitened matrix of the same shape as X
    """
    # Center the data: subtract the mean of each feature
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Construct the whitening matrix
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues+1e-10)) @ eigenvectors.T
    
    
    # Apply the whitening matrix to the centered data
    X_whitened = X_centered @ whitening_matrix
    
    return X_whitened


def make_gaussian_prob_vec(bin_vals, loc, scale, do_normalize=True):

    """
    create a gaussian probability distributioin at bin_vals,
    where the mean is loc, and standard deviation is scale

    if do_normalize, returns a normalized probability distribution that sums to 1.0
    """

    gauss_vec = norm.pdf(bin_vals, loc=loc, scale=scale)

    if do_normalize:
        return gauss_vec / gauss_vec.sum()

    else:
        return gauss_vec
