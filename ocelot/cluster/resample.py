"""Functions for re-sampling Gaia data within errors."""

import numpy as np
import pandas as pd


def generate_gaia_covariance_matrix(data_gaia: pd.DataFrame, type='pmra_pmdec_parallax') -> np.ndarray:
    """Generates a covariance matrix for Gaia data. Currently only has pmra/pmdec/parallax covariance matrix support,
    but could be easily extended in the future to also/or re-sample photometry.

    Notes:
        CHECKS ARE TURNED OFF TO MAKE THIS FUNCTION FASTER, so please make sure that data_gaia is correct and that
        no values have been messed up! No warranty is included on this BlazingFastTM function.

    Args:
        data_gaia (pd.DataFrame): Gaia data to make a covariance matrix for. Must contain the following keys:
            pmra_error
            pmdec_error
            parallax_error
            pmra_pmdec_corr
            parallax_pmra_corr
            parallax_pmdec_corr
            see Gaia release notes for help.
        type (str): what type of correlation matrix to return. Currently only supports 'pmra_pmdec_parallax'.
            Default: 'pmra_pmdec_parallax'

    Returns:
        if type == 'pmra_pmdec_parallax':
            a covariance matrix of shape (n_samples, 3, 3) for pmra, pmdec and parallax (in that order). See
            Cantat-Gaudin+2018a (the TGAS paper) equation 2.

    """
    if type != 'pmra-pmdec-parallax':
        raise NotImplementedError("currently, this function only supports pmra_pmdec_parallax covariance matrices.")

    # Grab the errors we need but as numpy arrays
    pmra = data_gaia['pmra_error'].to_numpy()
    pmdec = data_gaia['pmdec_error'].to_numpy()
    parallax = data_gaia['parallax_error'].to_numpy()

    # Also get the correlation coefficients, which come in the range [-1, 1]
    corr_pmra_pmdec = data_gaia['pmra_pmdec_corr'].to_numpy()
    corr_pmra_parallax = data_gaia['parallax_pmra_corr'].to_numpy()
    corr_pmdec_parallax = data_gaia['parallax_pmdec_corr'].to_numpy()

    # Calculate all of the terms
    pmra_pmra = pmra**2
    pmdec_pmdec = pmdec**2
    parallax_parallax = parallax**2

    pmra_pmdec = pmra * pmdec * corr_pmra_pmdec
    pmra_parallax = pmra * parallax * corr_pmra_parallax
    pmdec_parallax = pmdec * parallax * corr_pmdec_parallax

    # And finally, put it all into a big array (which we transpose from shape (3, 3, n_samples) to shape
    # (n_samples, 3, 3) to make later calculations more easily vectorisable)
    return np.asarray([[pmra_pmra,     pmra_pmdec,     pmra_parallax],
                       [pmra_pmdec,    pmdec_pmdec,    pmdec_parallax],
                       [pmra_parallax, pmdec_parallax, parallax_parallax]]).T


def resample_gaia_astrometry(data_gaia: pd.DataFrame, covariance_matrix: np.ndarray) -> pd.DataFrame:
    """ Todo fstring
    Notes:
        CHECKS ARE TURNED OFF TO MAKE THIS FUNCTION FASTER, so please make sure that data_gaia is correct and that
        no values have been messed up! No warranty is included on this BlazingFastTM function.
    """

    vectorized_multivariate_normal = np.vectorize(
        np.random.multivariate_normal, excluded='check_valid', cache=True, signature='(n),(n,n),()->(n)')

    return vectorized_multivariate_normal(mean=data_gaia[['pmra', 'pmdec', 'parallax']].to_numpy(),
                                          cov=covariance_matrix,
                                          check_valid=False)
