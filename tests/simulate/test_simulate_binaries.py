"""Some utilities to test the finer (and complicated) parts of binary star generation in
a simulated cluster.
"""

from ocelot.simulate import astrometry
from astropy import units as u
from astropy import constants
import numpy as np
import kepler 


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

    primary, secondary, _ = astrometry._compute_semimajor_axes(
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
    )[:2]
    samples = np.sum(samples, axis=0)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples,)
    np.testing.assert_allclose(
        samples.mean(), pluto_average_solar_distance, rtol=0.05, atol=0.0
    )


def _random_binaries(n_samples, seed=42):
    """Random binary parameters spanning circular to very eccentric orbits."""
    rng = np.random.default_rng(seed=seed)
    primary_mass = rng.uniform(0.5, 3.0, size=n_samples)
    mass_ratio = rng.uniform(0.1, 1.0, size=n_samples)
    period = 10 ** rng.uniform(0, 6, size=n_samples)  # days
    eccentricity = rng.uniform(0.0, 0.95, size=n_samples)
    mean_anomaly = rng.uniform(0, 2 * np.pi, size=n_samples)
    return (
        primary_mass,
        mass_ratio * primary_mass,
        mass_ratio,
        period,
        eccentricity,
        mean_anomaly,
    )


def _true_anomaly(mean_anomaly, eccentricity):
    """Cosine and sine of the true anomaly, computed from the eccentric anomaly.

    kepler.kepler also returns cos(f) and sin(f), but its sin(f) is rounded to zero
    within about 1e-5 rad of apoapsis, which breaks the numerical derivatives below.
    """
    eccentric_anomaly, _, _ = kepler.kepler(mean_anomaly, eccentricity)
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2),
    )
    return np.cos(true_anomaly), np.sin(true_anomaly)


def _orbital_plane_positions(
    secondary_mass, mass_ratio, period, eccentricity, mean_anomaly
):
    """Positions (in pc) of both stars in the orbital plane at a given mean anomaly."""
    primary_axis, secondary_axis, _ = astrometry._compute_semimajor_axes(
        secondary_mass, mass_ratio, period
    )
    cos_f, sin_f = _true_anomaly(mean_anomaly, eccentricity)
    direction = np.asarray([cos_f, sin_f, np.zeros_like(cos_f)]).T
    primary = (
        -astrometry._current_distance_from_barycentre(
            primary_axis, eccentricity, cos_f
        ).reshape(-1, 1)
        * direction
    )
    secondary = (
        astrometry._current_distance_from_barycentre(
            secondary_axis, eccentricity, cos_f
        ).reshape(-1, 1)
        * direction
    )
    return primary, secondary


def test_current_orbital_velocities_physics():
    """Checks orbital velocities against momentum conservation, the vis-viva equation
    and numerical derivatives of the orbital positions.
    """
    n_samples = 10000
    primary_mass, secondary_mass, mass_ratio, period, eccentricity, mean_anomaly = (
        _random_binaries(n_samples)
    )
    _, _, semimajor_axis = astrometry._compute_semimajor_axes(
        secondary_mass, mass_ratio, period
    )
    cos_f, sin_f = _true_anomaly(mean_anomaly, eccentricity)
    velocity_primary, velocity_secondary = astrometry._current_orbital_velocities(
        primary_mass, secondary_mass, semimajor_axis, eccentricity, cos_f, sin_f
    )
    assert velocity_primary.shape == (n_samples, 3)
    assert velocity_secondary.shape == (n_samples, 3)

    # Orbits are flat in this frame
    np.testing.assert_array_equal(velocity_primary[:, 2], 0.0)

    # Total momentum about the barycentre is zero
    momentum = (
        primary_mass.reshape(-1, 1) * velocity_primary
        + secondary_mass.reshape(-1, 1) * velocity_secondary
    )
    np.testing.assert_allclose(
        np.linalg.norm(momentum, axis=1)
        / (primary_mass * np.linalg.norm(velocity_primary, axis=1)),
        0.0,
        atol=1e-12,
    )

    # Relative speed obeys the vis-viva equation
    primary, secondary = _orbital_plane_positions(
        secondary_mass, mass_ratio, period, eccentricity, mean_anomaly
    )
    separation = np.linalg.norm(secondary - primary, axis=1)
    expected_speed = (
        np.sqrt(
            constants.G
            * (primary_mass + secondary_mass)
            * u.M_sun
            * (2 / (separation * u.pc) - 1 / (semimajor_axis * u.pc))
        )
        .to(u.m / u.s)
        .value
    )
    np.testing.assert_allclose(
        np.linalg.norm(velocity_secondary - velocity_primary, axis=1),
        expected_speed,
        rtol=1e-10,
    )

    # Velocities are the time derivative of the positions (central differences)
    step = 1e-7  # in mean anomaly
    time_step = (step / (2 * np.pi) * period * u.day).reshape(-1, 1)
    primary_ahead, secondary_ahead = _orbital_plane_positions(
        secondary_mass, mass_ratio, period, eccentricity, mean_anomaly + step
    )
    primary_behind, secondary_behind = _orbital_plane_positions(
        secondary_mass, mass_ratio, period, eccentricity, mean_anomaly - step
    )
    for velocity, ahead, behind in (
        (velocity_primary, primary_ahead, primary_behind),
        (velocity_secondary, secondary_ahead, secondary_behind),
    ):
        numerical = ((ahead - behind) * u.pc / (2 * time_step)).to(u.m / u.s).value
        np.testing.assert_allclose(
            np.linalg.norm(numerical - velocity, axis=1)
            / np.linalg.norm(velocity, axis=1),
            0.0,
            atol=1e-5,
        )