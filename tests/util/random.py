from ocelot.util.random import points_on_sphere


import matplotlib.pyplot as plt
import numpy as np


def test_points_on_sphere(show_diagnostic_histograms=False):
    """A function for testing the random spherical co-ordinates generator."""
    np.random.seed(42)

    # Calls and a basic diagnostic plot if requested
    theta_rad, phi_rad = points_on_sphere(
        3000, radians=True, phi_symmetric=False
    )
    theta_deg, phi_deg = points_on_sphere(
        3000, radians=False, phi_symmetric=True
    )

    if show_diagnostic_histograms:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].hist(theta_deg, bins="auto")
        ax[0].set_title("distribution of theta in degrees")
        ax[1].hist(phi_deg, bins="auto")
        ax[1].set_title("distribution of phi in degrees")
        fig.show()
        plt.close("all")

    # Check radian input_mode, asymmetric
    assert np.all(np.logical_and(theta_rad >= 0, theta_rad < 2 * np.pi))
    assert np.all(np.logical_and(phi_rad >= 0, phi_rad <= np.pi))

    # Check degree input_mode, symmetric
    assert np.all(np.logical_and(theta_deg >= 0, theta_deg < 360))
    assert np.all(np.logical_and(phi_deg >= -90, phi_deg <= 90))

    # Also check that phi appears correctly distributed as it's the hard one here
    assert np.allclose(np.std(phi_deg), 39.22, atol=0.5)