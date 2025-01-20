"""Some utilities to test the finer (and complicated) parts of binary star generation in
a simulated cluster.
"""

from ocelot.simulate import astrometry
from astropy import units as u
from astropy import constants
import numpy as np


pluto_mass = (869.3 * u.km**3 / u.s**2 / constants.G).to(u.M_sun).value
charon_mass = (106.1 * u.km**3 / u.s**2 / constants.G).to(u.M_sun).value
charon_period = 6.387221  # days

charon_planetocentric_semimajor_axis = (19595.764 * u.km).to(u.pc).value
charon_barycentric_semimajor_axis = (17181.0 * u.km).to(u.pc).value

# https://www.universetoday.com/44534/plutos-distance-from-the-sun/
# yes this is a reputable source for my unit test
pluto_average_solar_distance = (39.48168677 * u.au).to(u.pc).value

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/plutofact.html
pluto_orbital_period = 90560
pluto_orbital_eccentricity = 0.2444


def test_kepler_third_law():
    # Plug in some numbers for Earth
    earth_semimajor_axis = (
        (astrometry._semimajor_axis(1.0, 365.256363) * u.pc).to(u.au).value
    )
    np.testing.assert_allclose(earth_semimajor_axis, 1.0, rtol=0.0, atol=1e-6)


def test_semimajor_axis():
    """Tests ability to determine barycentric & total semimajor axes. Uses values for
    the Pluto/Charon system taken from
    https://ui.adsabs.harvard.edu/abs/2024AJ....167..256B/abstract
    """
    total_mass = pluto_mass + charon_mass
    semimajor_axis = astrometry._semimajor_axis(total_mass, charon_period)
    np.testing.assert_allclose(
        semimajor_axis, charon_planetocentric_semimajor_axis, atol=0.0, rtol=1e-4
    )


def test_compute_semimajor_axes():
    """Tests ability to determine barycentric & total semimajor axes. Uses values for
    the Pluto/Charon system taken from
    https://ui.adsabs.harvard.edu/abs/2024AJ....167..256B/abstract
    """
    mass_ratio = charon_mass / pluto_mass

    primary, secondary = astrometry._compute_semimajor_axes(
        charon_mass, mass_ratio, charon_period
    )
    np.testing.assert_allclose(
        secondary, charon_barycentric_semimajor_axis, atol=0.0, rtol=2e-2
    )
    np.testing.assert_allclose(
        charon_planetocentric_semimajor_axis - charon_barycentric_semimajor_axis,
        primary,
        atol=0.0,
        rtol=2e-1,
    )


def test_compute_separation():
    """Tries to compute the mean separation of Pluto from the Sun to check everything
    works ok.
    """
    n_samples = 10000
    rng = np.random.default_rng(seed=42)
    samples = astrometry._compute_separation(
        np.repeat(pluto_mass, n_samples),
        pluto_mass,
        pluto_orbital_period,
        pluto_orbital_eccentricity,
        rng,
    )
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples,)
    np.testing.assert_allclose(
        samples.mean(), pluto_average_solar_distance, rtol=0.05, atol=0.0
    )
