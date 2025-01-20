"""Functions for dealing with astrometry-related things."""

from __future__ import annotations  # Necessary to type hint without cyclic import
import numpy as np
from ocelot.util.random import unit_vectors
import ocelot.simulate.cluster
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from astropy import constants
from scipy.stats import multivariate_normal
import kepler


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
    n_primaries = np.sum(primary)
    n_secondaries = len(cluster.cluster) - n_primaries

    # Firstly, make host star positions
    cluster.cluster[["x", "y", "z"]] = np.nan  # Todo see if can remove
    cluster.cluster.loc[primary, ["x", "y", "z"]] = cluster.models.distribution.rvs(
        n_primaries, seed=cluster.random_generator
    ).to(u.pc).value

    # Pull everything we might need out of the dataframe. This will make life easier
    host_x, host_y, host_z = (
        cluster.cluster.loc[
            cluster.cluster.loc[secondary, "index_primary"].to_numpy(), ["x", "y", "z"]
        ]
        .to_numpy()
        .T
    )
    secondary_mass, mass_ratio, period, eccentricity = (
        cluster.cluster.loc[secondary, ["mass", "mass_ratio", "period", "eccentricity"]]
        .to_numpy()
        .T
    )

    # Do some calculations for the binary stars
    separation = _compute_separation(
        secondary_mass, mass_ratio, period, eccentricity, cluster.random_generator
    )

    # Project the separation into a random direction
    x_unit_vector, y_unit_vector, z_unit_vector = unit_vectors(
        n_secondaries, seed=cluster.random_generator
    ).T
    cluster.cluster.loc[secondary, "x"] = host_x + x_unit_vector * separation
    cluster.cluster.loc[secondary, "y"] = host_y + y_unit_vector * separation
    cluster.cluster.loc[secondary, "z"] = host_z + z_unit_vector * separation

    # Check that nothing went horribly wrong
    if not np.all(np.isfinite(cluster.cluster[["x", "y", "z"]])):
        raise RuntimeError(
            "Something went wrong! At least one star has a non-finite position."
        )

    # Save some optional other things
    # cluster.cluster.loc[secondary, 'semimajor_axis_total'] = total_semimajor_axis
    # cluster.cluster.loc[secondary, 'semimajor_axis_primary'] = primary_semimajor_axis
    # cluster.cluster.loc[secondary, 'semimajor_axis_secondary'] = secondary_semimajor_axis
    # cluster.cluster.loc[secondary, 'orbit_mean_anomaly'] = mean_anomaly
    # cluster.cluster.loc[secondary, 'orbit_unit_vec_x'] = x_unit_vector
    # cluster.cluster.loc[secondary, 'orbit_unit_vec_y'] = y_unit_vector
    # cluster.cluster.loc[secondary, 'orbit_unit_vec_z'] = z_unit_vector
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
    primary_semimajor_axis, secondary_semimajor_axis = _compute_semimajor_axes(
        secondary_mass, mass_ratio, period
    )

    # Sample a mean anomaly & compute true anomaly
    cosine_of_true_anomaly = _sample_true_anomaly(secondary_mass, eccentricity, rng)

    # Calculate current positions of stars
    primary_radius = _current_distance_from_barycentre(
        primary_semimajor_axis, eccentricity, cosine_of_true_anomaly
    )
    secondary_radius = _current_distance_from_barycentre(
        secondary_semimajor_axis, eccentricity, cosine_of_true_anomaly
    )
    separation = primary_radius + secondary_radius
    return separation


def _sample_true_anomaly(secondary_mass, eccentricity, rng):
    """Samples (the cosine of) an orbit's true anomaly."""
    mean_anomaly = rng.uniform(0, np.pi * 2, size=len(secondary_mass))
    cosine_of_true_anomaly = kepler.kepler(mean_anomaly, eccentricity)[1]
    return cosine_of_true_anomaly


def _compute_semimajor_axes(secondary_mass, mass_ratio, period):
    """Computes the semimajor axis of the primary star's orbit and the secondary
    star's orbit.
    """
    primary_mass = secondary_mass / mass_ratio
    total_mass = primary_mass + secondary_mass

    total_semimajor_axis = _semimajor_axis(total_mass, period)

    primary_semimajor_axis = total_semimajor_axis * secondary_mass / total_mass
    secondary_semimajor_axis = total_semimajor_axis * primary_mass / total_mass

    return primary_semimajor_axis, secondary_semimajor_axis


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

    return CartesianDifferential(d_x=v_x, d_y=v_y, d_z=v_z, unit=u.m / u.s)


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
