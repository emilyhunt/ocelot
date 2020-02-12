"""Functions for calculating various dust-related things."""

import numpy as np
import pandas as pd
from typing import Union


# Wavelengths of the Gaia DR2 photometry in nanometres
gaia_dr2_wavelengths = {
    "G": 643.770,
    "G_BP": 530.957,
    "G_RP": 770.985
}

# A_lambda over A_V for Gaia DR2
gaia_dr2_a_lambda_over_a_v = {
    "G": 0.85926,
    "G_BP": 1.06794,
    "G_RP": 0.65199
}


def ccm_extinction_law(extinction_v: Union[np.ndarray, pd.Series, list, tuple, float, int],
                       wavelength: Union[float, int],
                       r_v: float = 3.1):
    """Calculates extinction values at a given wavelength using the CCM+1989 extinction law.

    Uses:
        A_lambda = A_V * (a(x) + b(x) / R_v)

    Args:
        extinction_v (list-like, float or int): existing V-band extinction in units of magnitude.
        wavelength (float, int): wavelength to calculate new constants at, in nanometres.
        r_v (float): CCM extinction law constant.
            Default: 3.1

    Returns:
        np.array of extinction_v evaluated at the specified wavelength.

    """
    # Cast everything as a numpy array
    extinction_v = np.asarray(extinction_v).reshape(-1)

    # Convert wavelength (nm) to inverse wavelength (microm^-1)
    x = 1e-3 / wavelength

    # Calculate stuff we need, in...
    # IR:
    if 0.3 <= x < 1.1:
        a = 0.574*x**1.61
        b = -0.527*x**1.61

    # Optical/NIR
    elif 1.1 <= x < 3.3:
        y = x - 1.82
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

    else:
        raise NotImplementedError(f"Sorry! Extinction at the requested wavelength of {wavelength:.1f}nm is not "
                                  f"implemented. Emily was too lazy to keep typing after doing IR and opt/NIR...")

    return extinction_v * (a + b / r_v)
