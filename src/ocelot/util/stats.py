"""Various simple statistical utility functions."""

import numpy as np

from typing import Literal


def variable_bin_histogram(values, min, max, minimum_width, minimum_size=5):
    """Computes a variably binned histogram."""
    if minimum_size * 2 >= len(values):
        raise ValueError(
            "minimum_size must be no more than half the size of values to create a "
            "histogram with at least two bins. However, values is only "
            f"{len(values)} long, compared to minimum_size which is {minimum_size}."
            "Consider increasing the size of your input dataset or reducing "
            "minimum_size."
        )

    # First pass
    n_bins = int(np.round((max - min) / minimum_width)) + 1
    bins = np.linspace(min, max, num=n_bins)

    count, _ = np.histogram(values, bins=bins)

    # Early return condition if all bins are fine / minimum_size is zero
    if minimum_size == 0 or np.all(count >= minimum_size):
        return count, bins

    # Error check that should stop any bins from ever not being filled
    if np.sum(count) < minimum_size:
        raise ValueError(
            "Unable to fill bins due to fewer than minimum_size items in total"
        )

    # Otherwise, loop over all values, removing items until we get the desired bin
    # occupancies
    index = 0
    bins, count = list(bins), list(count)
    while index < len(count) - 2:
        if count[index] < minimum_size:
            count[index] += count[index + 1]
            count.pop(index + 1)
            bins.pop(index + 1)
        else:
            index += 1

    # Handle the final bin as a special case (since we have to go in reverse)
    while count[-1] < minimum_size:
        index = len(count) - 1
        count[index] += count[index - 1]
        count.pop(index - 1)
        bins.pop(index - 1)

    return np.asarray(count), np.asarray(bins)


def calculate_bin_centers(bin_edges):
    """Calculates the centers of a binned histogram."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def vectorized_multivariate_normal_pdf(
    values: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Calculate the PDF of n_queries different multivariate normal distributions at
    n_queries different points in a fast and vectorized way. Significantly faster than
    scipy.stats.multivariate_normal.pdf() when n_queries is large.

    Parameters
    ----------
    values : np.ndarray
        Values to query the PDF at. Must have shape (n_queries, n_dims).
    means : np.ndarray
        Means of the distributions. Must have shape (n_queries, n_dims).
    covariances : np.ndarray
        Covariances of the distributions. Must have shape (n_queries, n_dims, n_dims).

    Returns
    -------
    np.ndarray
        Array of shape (n_queries,) giving the PDF value of each distribution.
    """
    k = means.shape[1]

    # Calculate products from covariances
    means = means.reshape(-1, k, 1)
    values = values.reshape(-1, k, 1)
    covariances = covariances.reshape(-1, k, k)
    determinants = np.linalg.det(covariances)
    inverses = np.linalg.inv(covariances)

    diff = values - means

    constant = (2 * np.pi) ** (-k / 2) * determinants ** (-0.5)
    exponent = -0.5 * (diff.reshape(-1, 1, k) @ inverses @ diff).flatten()

    return constant.flatten() * np.exp(exponent.flatten())


def vectorized_multivariate_normal_rvs(
    means: np.ndarray,
    covariances: np.ndarray,
    n_samples: int = 1,
    seed=None,
    method: Literal["cholesky"] = "cholesky",
) -> np.ndarray:
    """Sample n_samples samples from arrays of different multivariate normal
    distributions at n_samples different points in a fast and vectorized way.
    Significantly faster than scipy.stats.multivariate_normal.rvs() when n_queries is
    large.

    Parameters
    ----------
    means : np.ndarray
        Means of the distributions. Must have shape (n_dists, n_dims).
    covariances : np.ndarray
        Covariances of the distributions. Must have shape (n_dists, n_dims, n_dims).
    n_samples : int
        Number of samples to draw from each distribution. Default: 1
    method : { 'svd', 'eigh', 'cholesky'}, optional
        TODO: Only Cholesky is supported at this time.
        Method to use for matrix decompositions. From the numpy docs:
        "The cov input is used to compute a factor matrix A such that A @ A.T = cov. This
        argument is used to select the method used to compute the factor matrix A. The
        default method 'svd' is the slowest, while 'cholesky' is the fastest but less 
        robust than the slowest method. The method eigh uses eigen decomposition to 
        compute A and is faster than svd but slower than cholesky." Default: 'svd'

    Returns
    -------
    np.ndarray
        Array of samples of shape (n_samples, n_dists, n_dims).
    """
    k = means.shape[1]

    rng = np.random.default_rng(seed)

    # Calculate products from covariances
    means = means.reshape(-1, k)
    covariances = covariances.reshape(-1, k, k)

    shape = (n_samples, means.shape[0], k)
    X = rng.standard_normal(shape + (1,))
    
    match method:
        # case "svd":
        #     L = np.linalg.svd(covariances)
        # case "eigh":
        #     L = np.linalg.eigh(covariances)
        case "cholesky":
            L = np.linalg.cholesky(covariances)
        case _:
            raise ValueError(f"Specified method '{method}' not supported.")

    return (L @ X).reshape(shape) + means
