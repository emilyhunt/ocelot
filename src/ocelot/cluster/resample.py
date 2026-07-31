"""Functions for re-sampling Gaia data within errors."""

import numpy as np
import pandas as pd
from typing import Sequence, Literal
from ocelot.util.stats import vectorized_multivariate_normal_rvs


def resample_gaia_astrometry(
    data_gaia: pd.DataFrame,
    n_resamples: int = 1,
    suffixes: Sequence[str] | None = None,
    method: Literal["cholesky"] = "cholesky",
    include_ra_dec: bool = False,
    seed=None,
) -> pd.DataFrame:
    """Resample Gaia astrometric parameters for pmra, pmdec and parallax, given input
    best estimate means and covariance matrices.

    Parameters
    ----------
    data_gaia : pd.DataFrame
        data for the field, including keys 'astrometric_params_solved', 'pmra', 'pmdec',
        'parallax', 'pmra_error', 'pmdec_error', 'parallax_error', 'pmra_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pseudocolour',
        'pseudocolour_error', 'pmra_pseudocolour_corr', 'pmdec_pseudocolour_corr',
        'parallax_pseudocolour_corr'
    n_resamples : int, optional
        Number of resamples to perform. Default: 1
    suffixes : Iterable[str] | None, optional
        Suffixes to apply in the output dataframe's pmra, pmdec, and parallax columns
        per-resample.
    method : { 'svd', 'eigh', 'cholesky'}, optional
        TODO: Only Cholesky is supported at this time.
        Method to use for matrix decompositions. From the numpy docs:
        "The cov input is used to compute a factor matrix A such that A @ A.T = cov. This
        argument is used to select the method used to compute the factor matrix A. The
        default method 'svd' is the slowest, while 'cholesky' is the fastest but less
        robust than the slowest method. The method eigh uses eigen decomposition to
        compute A and is faster than svd but slower than cholesky." Default: 'svd'
    include_ra_dec : bool, optional
        When True, also include ra and dec uncertainties. Requires even more columns!
        Will be more accurate. Default: False
    seed : optional
        Seed for the random number generator. Default: None

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

    # Then do each solution
    start_dim = 0
    columns = ["pmra", "pmdec", "parallax"]
    if include_ra_dec:
        start_dim = 2
        data_gaia['ra_mas'] = data_gaia['ra'] * 60**2 * 1000
        data_gaia['dec_mas'] = data_gaia['dec'] * 60**2 * 1000

        columns = ["ra_mas", "dec_mas"] + columns

    resampled_astrometry = np.full((n_resamples, len(data_gaia), 3), np.nan)

    # 5 parameter resampling - "straightforward"
    if np.any(five_parameter_astrometry):
        covariance_matrix_five = generate_gaia_covariance_matrix(
            data_gaia_five, include_ra_dec=include_ra_dec
        )

        resampled_astrometry[:, five_parameter_astrometry] = (
            vectorized_multivariate_normal_rvs(
                means=data_gaia_five[columns].to_numpy(),
                covariances=covariance_matrix_five,
                n_samples=n_resamples,
                method=method,
                seed=seed,
            )[:, :, start_dim:]
        )

    # 6 parameter resampling - we drop colours
    if np.any(six_parameter_astrometry):
        covariance_matrix_six = generate_gaia_covariance_matrix(
            data_gaia_six, six_parameter_sources=True, include_ra_dec=include_ra_dec
        )

        resampled_astrometry[:, six_parameter_astrometry] = (
            vectorized_multivariate_normal_rvs(
                means=data_gaia_six[columns + ["pseudocolour"]].to_numpy(),
                covariances=covariance_matrix_six,
                n_samples=n_resamples,
                method=method,
                seed=seed,
            )[:, :, start_dim:-1]
        )

    # Reshape the resampled astrometry into a dict, getting ready to turn it into a df
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
    data_gaia: pd.DataFrame,
    six_parameter_sources: bool = False,
    include_ra_dec: bool = True,
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
            pseudocolour, pseudocolour_error, pmra_pseudocolour_corr,
            pmdec_pseudocolour_corr, parallax_pseudocolour_corr
        see Gaia release notes for help.
    six_parameter_sources : bool, optional
        Whether or not ALL sources in data_gaia also depend on the estimated
        pseudocolour. If true, will return matrices of shape (n_samples, 4, 4) instead.
        Default: False
    include_ra_dec : bool, optional
        When True, also include ra and dec uncertainties, generating an
        (n_samples, 5, 5) or (n_samples, 6, 6) (six_parameter_sources=True) shape-output
        instead. Default: True

    Returns
    -------
    np.ndarray
        Array of covariance matrices of shape (n_stars, n_params, n_params). n_params
        is 4 if six_parameter_sources is true, otherwise it is 3. include_ra_dec being
        True adds 2 to this.
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
    plx_plx = parallax**2

    pmra_pmdec = pmra * pmdec * corr_pmra_pmdec
    pmra_plx = pmra * parallax * corr_pmra_parallax
    pmdec_plx = pmdec * parallax * corr_pmdec_parallax

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
        plx_color = parallax * color * corr_parallax_color

        if not include_ra_dec:
            return np.asarray(
                [
                    [pmra_pmra, pmra_pmdec, pmra_plx, pmra_color],
                    [pmra_pmdec, pmdec_pmdec, pmdec_plx, pmdec_color],
                    [pmra_plx, pmdec_plx, plx_plx, plx_color],
                    [pmra_color, pmdec_color, plx_color, color_color],
                ]
            ).T

        (
            ra_ra,
            dec_dec,
            ra_dec,
            ra_pmra,
            ra_pmdec,
            ra_plx,
            dec_pmra,
            dec_pmdec,
            dec_plx,
        ) = _get_ra_dec_terms(data_gaia, pmra, pmdec, parallax)

        corr_ra_color = data_gaia["ra_pseudocolour_corr"].to_numpy()
        corr_dec_color = data_gaia["dec_pseudocolour_corr"].to_numpy()

        ra_color = pmra * color * corr_ra_color
        dec_color = pmdec * color * corr_dec_color

        return np.asarray(
            [
                [ra_ra, ra_dec, ra_pmra, ra_pmdec, ra_plx, ra_color],
                [ra_dec, dec_dec, dec_pmra, dec_pmdec, dec_plx, dec_color],
                [ra_pmra, dec_pmra, pmra_pmra, pmra_pmdec, pmra_plx, pmra_color],
                [ra_pmdec, dec_pmdec, pmra_pmdec, pmdec_pmdec, pmdec_plx, pmdec_color],
                [ra_plx, dec_plx, pmra_plx, pmdec_plx, plx_plx, plx_color],
                [ra_color, dec_color, pmra_color, pmdec_color, plx_color, color_color],
            ]
        ).T

    if include_ra_dec:
        (
            ra_ra,
            dec_dec,
            ra_dec,
            ra_pmra,
            ra_pmdec,
            ra_plx,
            dec_pmra,
            dec_pmdec,
            dec_plx,
        ) = _get_ra_dec_terms(data_gaia, pmra, pmdec, parallax)
        return np.asarray(
            [
                [ra_ra, ra_dec, ra_pmra, ra_pmdec, ra_plx],
                [ra_dec, dec_dec, dec_pmra, dec_pmdec, dec_plx],
                [ra_pmra, dec_pmra, pmra_pmra, pmra_pmdec, pmra_plx],
                [ra_pmdec, dec_pmdec, pmra_pmdec, pmdec_pmdec, pmdec_plx],
                [ra_plx, dec_plx, pmra_plx, pmdec_plx, plx_plx],
            ]
        ).T

    return np.asarray(
        [
            [pmra_pmra, pmra_pmdec, pmra_plx],
            [pmra_pmdec, pmdec_pmdec, pmdec_plx],
            [pmra_plx, pmdec_plx, plx_plx],
        ]
    ).T


def _get_ra_dec_terms(data_gaia, pmra, pmdec, parallax):
    ra = data_gaia["ra_error"].to_numpy()
    dec = data_gaia["dec_error"].to_numpy()
    corr_ra_dec = data_gaia["ra_dec_corr"].to_numpy()
    corr_ra_pmra = data_gaia["ra_pmra_corr"].to_numpy()
    corr_ra_pmdec = data_gaia["ra_pmdec_corr"].to_numpy()
    corr_ra_parallax = data_gaia["ra_parallax_corr"].to_numpy()
    corr_dec_pmra = data_gaia["dec_pmra_corr"].to_numpy()
    corr_dec_pmdec = data_gaia["dec_pmdec_corr"].to_numpy()
    corr_dec_parallax = data_gaia["dec_parallax_corr"].to_numpy()

    ra_ra = ra**2
    dec_dec = dec**2

    ra_dec = ra * dec * corr_ra_dec
    ra_pmra = ra * pmra * corr_ra_pmra
    ra_pmdec = ra * pmdec * corr_ra_pmdec
    ra_plx = ra * parallax * corr_ra_parallax

    dec_pmra = dec * pmra * corr_dec_pmra
    dec_pmdec = dec * pmdec * corr_dec_pmdec
    dec_plx = dec * parallax * corr_dec_parallax
    return (
        ra_ra,
        dec_dec,
        ra_dec,
        ra_pmra,
        ra_pmdec,
        ra_plx,
        dec_pmra,
        dec_pmdec,
        dec_plx,
    )
