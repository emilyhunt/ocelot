"""Functions for dealing with astrometry-related things."""

from __future__ import annotations  # Necessary to type hint without cyclic import

import kepler
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, SkyCoord
from scipy.stats import multivariate_normal

import ocelot.simulate.cluster
from ocelot.util.random import rotation_matrix


def generate_star_positions(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Generates positions of member stars in polar coordinates relative to the center
    of the cluster.
    """
    # Also handle binary star positions
    if cluster.features.binary_stars:
        return CartesianRepresentation(
            *generate_star_positions_with_binaries(cluster), unit=u.pc
        )
    return CartesianRepresentation(
        *cluster.models.distribution.rvs(
            len(cluster.cluster), seed=cluster.random_generator
        ).T,
    )


def generate_star_positions_with_binaries(
    cluster: ocelot.simulate.cluster.SimulatedCluster,
):
    """Generates locations relative to their host star for secondaries in the cluster.

    Uses some help from https://space.stackexchange.com/questions/8911/determining-orbital-position-at-a-future-point-in-time
    As well as diagram from https://physics.stackexchange.com/questions/61116/semi-major-axis-and-ellipticity-of-a-binary-system
    """
    # Firstly, let's make a temporary dataframe to store parameters in. This is easier
    # as we need to do a LOT of indexing.
    primary = (cluster.cluster["index_primary"] == -1).to_numpy()
    secondary = np.invert(primary)
    index_primary = cluster.cluster.loc[secondary, "index_primary"].to_numpy()
    n_primaries = np.sum(primary)
    n_secondaries = len(cluster.cluster) - n_primaries

    # Firstly, make host star positions
    cluster.cluster[["x", "y", "z"]] = np.nan  # Todo see if can remove

    # TODO should be using the barycenter position per-primary, and not offsetting the secondary based on the primary. This will cause particular issues for triples or wide binaries.
    cluster.cluster.loc[primary, ["x", "y", "z"]] = (
        cluster.models.distribution.rvs(n_primaries, seed=cluster.random_generator)
        .to(u.pc)
        .value
    )

    # Pull everything we might need out of the dataframe. This will make life easier
    host_x, host_y, host_z = (
        cluster.cluster.loc[index_primary, ["x", "y", "z"]].to_numpy().T
    )
    secondary_mass, mass_ratio, period, eccentricity = (
        cluster.cluster.loc[secondary, ["mass", "mass_ratio", "period", "eccentricity"]]
        .to_numpy()
        .T
    )

    # Do some calculations for the binary stars
    # Firstly, grab all of their positional information
    # TODO this return is super ugly. Can I split these functions in a nicer way?
    (
        primary_radius,
        secondary_radius,
        primary_semimajor_axis,
        secondary_semimajor_axis,
        total_semimajor_axis,
        mean_anomaly,
        eccentric_anomaly,
        cos_true_anomaly,
        sin_true_anomaly,
    ) = _compute_separation(
        secondary_mass, mass_ratio, period, eccentricity, cluster.random_generator
    )
    total_separation = primary_radius + secondary_radius

    # Secondly - while we're here - we can also grab current orbital speeds at these
    # positions, which may be used later to calculate other things
    total_mass = secondary_mass / mass_ratio + secondary_mass
    primary_speed = _current_orbital_velocity(
        total_mass, primary_semimajor_axis, primary_radius
    )
    secondary_speed = _current_orbital_velocity(
        total_mass, secondary_semimajor_axis, secondary_radius
    )

    # Project the separation into a random direction
    rotation_matrices = rotation_matrix(n_secondaries, seed=cluster.random_generator)

    # Firstly, grab location of primary & secondary in a flat, 2D plane
    barycenter_location = np.asarray([host_x, host_y, host_z]).T
    z = np.zeros_like(cos_true_anomaly)
    primary_2d = (
        primary_radius * np.asarray([-cos_true_anomaly, -sin_true_anomaly, z]).T
    )
    secondary_2d = (
        secondary_radius * np.asarray([cos_true_anomaly, sin_true_anomaly, z]).T
    )
    primary_location = rotation_matrices @ primary_2d
    secondary_location = rotation_matrices @ secondary_2d + barycenter_location

    cluster.cluster.loc[secondary, "x"] = secondary_location[:, 0]
    cluster.cluster.loc[secondary, "y"] = secondary_location[:, 1]
    cluster.cluster.loc[secondary, "z"] = secondary_location[:, 2]
    # TODO this may not work with triples/higher-order multiples
    cluster.cluster.loc[index_primary, "x"] += primary_location[:, 0]
    cluster.cluster.loc[index_primary, "y"] += primary_location[:, 1]
    cluster.cluster.loc[index_primary, "z"] += primary_location[:, 2]

    # Check that nothing went horribly wrong
    if not np.all(np.isfinite(cluster.cluster[["x", "y", "z"]])):
        raise RuntimeError(
            "Something went wrong! At least one star has a non-finite position."
        )

    # Save some optional other things which we'll need later to compute things
    cluster.cluster.loc[secondary, "semimajor_axis"] = total_semimajor_axis
    cluster.cluster.loc[secondary, "semimajor_axis_primary"] = primary_semimajor_axis
    cluster.cluster.loc[secondary, "semimajor_axis_secondary"] = (
        secondary_semimajor_axis
    )
    cluster.cluster.loc[secondary, "orbit_radius_primary"] = primary_radius
    cluster.cluster.loc[secondary, "orbit_radius_secondary"] = secondary_radius
    cluster.cluster.loc[secondary, "orbit_mean_anomaly"] = mean_anomaly
    cluster.cluster.loc[secondary, "orbit_eccentric_anomaly"] = eccentric_anomaly
    cluster.cluster.loc[secondary, "orbit_true_anomaly"] = np.arccos(cos_true_anomaly)
    cluster.cluster.loc[secondary, "orbit_time_periapsis"] = (
        cluster.parameters.epoch
        + mean_anomaly / 2 * np.pi * (period * u.day).to(u.yr).value
    )
    cluster.cluster.loc[secondary, "orbit_unit_vec_x"] = x_unit_vector
    cluster.cluster.loc[secondary, "orbit_unit_vec_y"] = y_unit_vector
    cluster.cluster.loc[secondary, "orbit_unit_vec_z"] = z_unit_vector

    cluster.cluster.loc[secondary, "orbit_speed_primary"] = primary_speed
    cluster.cluster.loc[secondary, "orbit_speed_secondary"] = secondary_speed
    # cluster.cluster.loc[secondary, 'orbit_separation'] = separation

    # Remove x/y/z columns else they'll just be confusing later! We hijacked the df!!!
    x, y, z = (
        cluster.cluster["x"].to_numpy().copy(),
        cluster.cluster["y"].to_numpy().copy(),
        cluster.cluster["z"].to_numpy().copy(),
    )
    cluster.cluster = cluster.cluster.drop(columns=["x", "y", "z"])
    return x, y, z


def _compute_separation(secondary_mass, mass_ratio, period, eccentricity, rng):
    """Computes the separation (in parsecs) between binary stars in a cluster."""
    primary_semimajor_axis, secondary_semimajor_axis, total_semimajor_axis = (
        _compute_semimajor_axes(secondary_mass, mass_ratio, period)
    )

    # Sample a mean anomaly & compute true anomaly
    eccentric_anomaly, cosine_of_true_anomaly, mean_anomaly = _sample_true_anomaly(
        secondary_mass, eccentricity, rng
    )

    # Calculate current positions of stars
    primary_radius = _current_distance_from_barycentre(
        primary_semimajor_axis, eccentricity, cosine_of_true_anomaly
    )
    secondary_radius = _current_distance_from_barycentre(
        secondary_semimajor_axis, eccentricity, cosine_of_true_anomaly
    )

    return (
        primary_radius,
        secondary_radius,
        primary_semimajor_axis,
        secondary_semimajor_axis,
        total_semimajor_axis,
        mean_anomaly,
        eccentric_anomaly,
        cosine_of_true_anomaly,
        sine_of_true_anomaly,
    )


def _sample_true_anomaly(secondary_mass, eccentricity, rng):
    """Samples (the cosine of) an orbit's true anomaly."""
    mean_anomaly = rng.uniform(0, np.pi * 2, size=len(secondary_mass))
    eccentric_anomaly, cosine_of_true_anomaly, sine_of_true_anomaly = kepler.kepler(
        mean_anomaly, eccentricity
    )
    return eccentric_anomaly, cosine_of_true_anomaly, sine_of_true_anomaly, mean_anomaly


def _compute_semimajor_axes(secondary_mass, mass_ratio, period):
    """Computes the semimajor axis of the primary star's orbit and the secondary
    star's orbit.
    """
    primary_mass = secondary_mass / mass_ratio
    total_mass = primary_mass + secondary_mass

    total_semimajor_axis = _semimajor_axis(total_mass, period)

    primary_semimajor_axis = total_semimajor_axis * secondary_mass / total_mass
    secondary_semimajor_axis = total_semimajor_axis * primary_mass / total_mass

    return primary_semimajor_axis, secondary_semimajor_axis, total_semimajor_axis


def _semimajor_axis(total_mass, period):
    """Kepler's 3rd law, arranged to give semi-major axis.

    Total mass should be in solar masses, and period should be in days.

    Returns semimajor axis in parsecs.
    """
    period_days = period << u.day
    mass_msun = total_mass << u.M_sun

    semimajor_axis = (
        (period_days) ** 2 * constants.G * (mass_msun) / (4 * np.pi**2)
    ) ** (1 / 3)
    return semimajor_axis.to(u.pc).value


def _current_distance_from_barycentre(
    semimajor_axis, eccentricity, cosine_of_true_anomaly
):
    """Computes the current distance of an orbiting body from the barycentre of a
    system.
    """
    return (
        semimajor_axis
        * (1 - eccentricity**2)
        / (1 + eccentricity * cosine_of_true_anomaly)
    )


def _current_orbital_velocity(
    total_mass,
    semimajor_axis,
    current_radius,
):
    """Computes the current orbital velocity of an object orbiting a body using the
    vis-viva equation.
    """
    return np.sqrt(
        constants.G
        * total_mass
        * u.Msun
        * (2 / (current_radius * u.pc) - 1 / (semimajor_axis * u.pc))
    )


def generate_star_velocities(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Generates the velocities of stars in a cluster."""
    # Todo should get velocities from the distribution object eventually
    distribution = multivariate_normal(
        mean=np.zeros(3),
        cov=cluster.parameters.velocity_dispersion_1d**2,
        seed=cluster.random_generator,
    )
    v_x, v_y, v_z = distribution.rvs(len(cluster.cluster)).T.reshape(
        3, -1
    )  # We also reshape to make sure a size-1 cluster is handled correctly

    # Leave early if we don't have binaries
    if not cluster.features.binary_stars:
        return CartesianDifferential(d_x=v_x, d_y=v_y, d_z=v_z, unit=u.m / u.s)

    # Otherwise, get to work making velocities that are offset
    secondary = (cluster.cluster["index_primary"] != -1).to_numpy()
    index_primaries = cluster.cluster.loc[secondary, "index_primary"].to_numpy()


def generate_true_star_astrometry(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Generates the true values of cluster astrometry (not affected by errors)."""
    positions = generate_star_positions(cluster)
    velocities = generate_star_velocities(cluster)

    # Do coordinate frame stuff to get final values (dont look astropy devs, dont tell
    # me what I can and cant do, i live in a lawless realm, this works so it works)
    cluster_center = cluster.parameters.position.transform_to(
        "galactocentric"
    ).cartesian
    cluster_differential = cluster_center.differentials["s"]

    final_positions = positions + cluster_center
    final_velocities = velocities + cluster_differential

    final_coords = SkyCoord(
        CartesianRepresentation(final_positions, differentials=final_velocities),
        frame="galactocentric",
    ).transform_to("icrs")
    final_coords_galactic = final_coords.transform_to("galactic")

    # Assign these values to cluster df
    cluster.cluster["ra"] = final_coords.ra.value
    cluster.cluster["dec"] = final_coords.dec.value
    cluster.cluster["l"] = final_coords_galactic.l.value
    cluster.cluster["b"] = final_coords_galactic.b.value
    cluster.cluster["pmra"] = final_coords.pm_ra_cosdec.value
    cluster.cluster["pmdec"] = final_coords.pm_dec.value
    cluster.cluster["parallax"] = 1000 / final_coords.distance.value
    cluster.cluster["pmra_true"] = cluster.cluster["pmra"]
    cluster.cluster["pmdec_true"] = cluster.cluster["pmdec"]
    cluster.cluster["parallax_true"] = cluster.cluster["parallax"]
    cluster.cluster["radial_velocity_true"] = final_coords.radial_velocity.value


# def generate_cluster_astrometry(cluster: ocelot.simulate.cluster.SimulatedCluster):
#     """Generates the astrometry of clusters."""
#     generate_true_star_astrometry(cluster)
#     apply_gaia_astrometric_uncertainties(cluster)
