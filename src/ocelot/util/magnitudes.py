import numpy as np


def add_two_magnitudes(magnitude_1, magnitude_2):
    """Correct (simplified) equation to add two magnitudes.
    Source: https://www.astro.keele.ac.uk/jkt/pubs/JKTeq-fluxsum.pdf
    """
    return -2.5 * np.log10(10 ** (-magnitude_1 / 2.5) + 10 ** (-magnitude_2 / 2.5))