import numpy as np


def add_two_magnitudes(
    magnitude_1: float | np.ndarray, magnitude_2: float | np.ndarray
):
    """Correct (simplified) equation to add two magnitudes.
    Source: https://www.astro.keele.ac.uk/jkt/pubs/JKTeq-fluxsum.pdf

    Parameters
    ----------
    magnitude_1 : float, np.ndarray
        First magnitude.
    magnitude_2 : float, np.ndarray
        First magnitude.

    Returns
    -------
    float, np.ndarray
        Summed magnitudes with same type as input.
    """
    return -2.5 * np.log10(10 ** (-magnitude_1 / 2.5) + 10 ** (-magnitude_2 / 2.5))
