"""Common functions to help with transforms and sampling."""

import numpy as np


def spherical_to_cartesian(radii, thetas, phis):
    """Converts from spherical to Cartesian coordinates. See:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Assumes radii ∈ [0, inf), thetas ∈ [0, pi], phis ∈ [0, 2pi)
    """
    x_values = radii * np.sin(thetas) * np.cos(phis)
    y_values = radii * np.sin(thetas) * np.sin(phis)
    z_values = radii * np.cos(thetas)
    return x_values, y_values, z_values
