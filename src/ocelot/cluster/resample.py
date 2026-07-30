"""Functions for re-sampling Gaia data within errors."""

import numpy as np
import pandas as pd
from typing import Iterable, Literal
from ocelot.util.stats import vectorized_multivariate_normal_rvs


def resample_gaia_astrometry(
    data_gaia: pd.DataFrame,
    n_resamples: int = 1,
    suffixes: Iterable[str] | None = None,
    method: Literal["svd", "eigh" "cholesky"] = "svd",
) -> pd.DataFrame:
    """Resample Gaia astrometric parameters for pmra, pmdec and parallax, given input
    best estimate means and covariance matrices.

    Parameters
    ----------
    data_gaia : pd.DataFrame
        data for the field, including keys 'astrometric_params_solved', 'pmra', 'pmdec',
        'parallax', 'pmra_error', 'pmdec_error', 'parallax_error', 'pmra_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pseudocolour_error',
        'pmra_pseudocolour_corr', 'pmdec_pseudocolour_corr',
        'parallax_pseudocolour_corr'
    n_resamples : int, optional
        Number of resamples to perform. Default: 1
    suffixes : Iterable[str] | None, optional
        Suffixes to apply in the output dataframe's pmra, pmdec, and parallax columns
        per-resample.
    method : { 'svd', 'eigh', 'cholesky'}, optional
        Method to use for matrix decompositions. From the numpy docs:
        "The cov input is used to compute a factor matrix A such that A @ A.T = cov. This
        argument is used to select the method used to compute the factor matrix A. The
        default method 'svd' is the slowest, while 'cholesky' is the fastest but less
        robust than the slowest method. The method eigh uses eigen decomposition to
        compute A and is faster than svd but slower than cholesky." Default: 'svd'

    Returns
    -------
    pd.DataFrame
        Output dataframe with same length as input, including 3*n_resamples columns
        containing resampled astrometry.

    Notes
    -----
    When using cholesky decomposition to speed this up, all covariance matrices must be
    positive definite. Basically, this means they should be symmetric and have no
    negative numbers. This is not always the case for Gaia covariance matrices; hence,
    the default method is the slower (but more robust) svd.
    """
    # TODO: requires a unit test
    # Firstly, let's identify which solutions are five or six parameter
    five_parameter_astrometry = data_gaia["astrometric_params_solved"].to_numpy() == 31
    six_parameter_astrometry = data_gaia["astrometric_params_solved"].to_numpy() == 95
    data_gaia_five = data_gaia.loc[five_parameter_astrometry]
    data_gaia_six = data_gaia.loc[six_parameter_astrometry]

    # Let's make somewhere to store solutions
    resampled_astrometry = np.zeros((n_resamples, len(data_gaia), 3))

    # 5 parameter resampling - "straightforward"
    if np.any(five_parameter_astrometry):
        covariance_matrix_five = generate_gaia_covariance_matrix(data_gaia_five)

        resampled_astrometry[:, five_parameter_astrometry] = (
            vectorized_multivariate_normal_rvs(
                mean=data_gaia_five[["pmra", "pmdec", "parallax"]].to_numpy(),
                cov=covariance_matrix_five,
                size=n_resamples,
                method=method,
            )
        )

    # 6 parameter resampling - we drop colours as the author of this script saw no need to use them =)
    if np.any(six_parameter_astrometry):
        covariance_matrix_six = generate_gaia_covariance_matrix(
            data_gaia_six, six_parameter_sources=True
        )

        resampled_astrometry[six_parameter_astrometry] = (
            vectorized_multivariate_normal_rvs(
                mean=data_gaia_six[
                    ["pmra", "pmdec", "parallax", "pseudocolour"]
                ].to_numpy(),
                cov=covariance_matrix_six,
                size=n_resamples,
                method=method,
            )[:, :, :-1]
        )

    # Reshape the resampled astrometry into a dict, getting ready to turn it into a dataframe
    if suffixes is None:
        suffixes = [f"_{x}" for x in range(n_resamples)]

    resampled_astrometry_dict = {}
    for i in range(n_resamples):
        resampled_astrometry_dict[f"pmra{suffixes[i]}"] = resampled_astrometry[i, :, 0]
        resampled_astrometry_dict[f"pmdec{suffixes[i]}"] = resampled_astrometry[i, :, 1]
        resampled_astrometry_dict[f"parallax{suffixes[i]}"] = resampled_astrometry[
            i, :, 2
        ]

    return pd.DataFrame(resampled_astrometry_dict)


def generate_gaia_covariance_matrix(
    data_gaia: pd.DataFrame, six_parameter_sources: bool = False
) -> np.ndarray:
    """Generates a covariance matrix for Gaia data. Currently only has pmra/pmdec/parallax(/color) covariance matrix
    support, but could be easily extended in the future to also/or re-sample more things.

    Parameters
    ----------
    data_gaia : pd.DataFrame
        Gaia data to make a covariance matrix for. Must contain the following keys:
            pmra_error, pmdec_error, parallax_error, pmra_pmdec_corr,
            parallax_pmra_corr, parallax_pmdec_corr
        For resampling for six parameter sources (i.e. six_parameter_sources=True),
        it must also contain:
            pseudocolour_error, pmra_pseudocolour_corr, pmdec_pseudocolour_corr,
            parallax_pseudocolour_corr
        see Gaia release notes for help.
    six_parameter_sources : bool, optional
        whether or not ALL sources in data_gaia also depend on the estimated
        pseudocolour. If true, will return matrices of shape (n_samples, 4, 4) instead.
        Default: False

    Returns
    -------
    np.ndarray
        Array of covariance matrices of shape (n_stars, n_params, n_params). n_params
        is 4 if six_parameter_sources is true, otherwise it is 3.
    """
    # TODO: requires a unit test
    # Grab the errors we need but as numpy arrays
    pmra = data_gaia["pmra_error"].to_numpy()
    pmdec = data_gaia["pmdec_error"].to_numpy()
    parallax = data_gaia["parallax_error"].to_numpy()

    # Also get the correlation coefficients, which come in the range [-1, 1]
    corr_pmra_pmdec = data_gaia["pmra_pmdec_corr"].to_numpy()
    corr_pmra_parallax = data_gaia["parallax_pmra_corr"].to_numpy()
    corr_pmdec_parallax = data_gaia["parallax_pmdec_corr"].to_numpy()

    # Calculate all of the terms
    pmra_pmra = pmra**2
    pmdec_pmdec = pmdec**2
    parallax_parallax = parallax**2

    pmra_pmdec = pmra * pmdec * corr_pmra_pmdec
    pmra_parallax = pmra * parallax * corr_pmra_parallax
    pmdec_parallax = pmdec * parallax * corr_pmdec_parallax

    # We have to do some extra steps if this source has a six-parameter solution (a DR3 thing).
    # Finally, we put it all into a big array (which we transpose from shape (3 or 4, 3 or 4, n_samples) to shape
    # (n_samples, 3 or 4, 3 or 4) to make later calculations more easily vectorisable)
    if six_parameter_sources:
        color = data_gaia["pseudocolour_error"].to_numpy()

        corr_pmra_color = data_gaia["pmra_pseudocolour_corr"].to_numpy()
        corr_pmdec_color = data_gaia["pmdec_pseudocolour_corr"].to_numpy()
        corr_parallax_color = data_gaia["parallax_pseudocolour_corr"].to_numpy()

        color_color = color**2

        pmra_color = pmra * color * corr_pmra_color
        pmdec_color = pmdec * color * corr_pmdec_color
        parallax_color = parallax * color * corr_parallax_color

        return np.asarray(
            [
                [pmra_pmra, pmra_pmdec, pmra_parallax, pmra_color],
                [pmra_pmdec, pmdec_pmdec, pmdec_parallax, pmdec_color],
                [pmra_parallax, pmdec_parallax, parallax_parallax, parallax_color],
                [pmra_color, pmdec_color, parallax_color, color_color],
            ]
        ).T

    return np.asarray(
        [
            [pmra_pmra, pmra_pmdec, pmra_parallax],
            [pmra_pmdec, pmdec_pmdec, pmdec_parallax],
            [pmra_parallax, pmdec_parallax, parallax_parallax],
        ]
    ).T
