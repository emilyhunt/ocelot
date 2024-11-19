"""Functions for dealing with astrometry-related things."""

from __future__ import annotations  # Necessary to type hint without cyclic import
import numpy as np
from ocelot.simulate.uncertainties import apply_gaia_astrometric_uncertainties
from ocelot.calculate.profile import sample_1d_king_profile
from ocelot.util.random import points_on_sphere
import ocelot.simulate.cluster
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from scipy.stats import multivariate_normal


def generate_star_positions(cluster: ocelot.simulate.cluster.Cluster):
    """Generates positions of member stars in polar coordinates relative to the center
    of the cluster.
    """
    radii = sample_1d_king_profile(
        cluster.parameters.r_core,
        cluster.parameters.r_tidal,
        cluster.stars,
        seed=cluster.random_seed,
    )
    phis, thetas = points_on_sphere(
        len(radii), phi_symmetric=False, seed=cluster.random_seed
    )
    x_values, y_values, z_values = spherical_to_cartesian(radii, thetas, phis)
    return CartesianRepresentation(x_values, y_values, z_values, unit=u.pc)


def spherical_to_cartesian(radii, thetas, phis):
    """Converts from spherical to Cartesian coordinates. See:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Assumes radii ∈ [0, inf), thetas ∈ [0, pi], phis ∈ [0, 2pi)
    """
    x_values = radii * np.sin(thetas) * np.cos(phis)
    y_values = radii * np.sin(thetas) * np.sin(phis)
    z_values = radii * np.cos(thetas)
    return x_values, y_values, z_values


def generate_star_velocities(cluster: ocelot.simulate.cluster.Cluster):
    """Generates the velocities of stars in a cluster."""
    distribution = multivariate_normal(
        mean=np.zeros(3),
        cov=np.identity(3) * cluster.parameters.velocity_dispersion_1d,
        seed=cluster.random_generator,
    )
    v_x, v_y, v_z = distribution.rvs(cluster.stars).T.reshape(3, -1)  # We also reshape to make sure a size-1 cluster is handled correctly
    return CartesianDifferential(d_x=v_x, d_y=v_y, d_z=v_z, unit=u.m / u.s)


def generate_true_star_astrometry(cluster: ocelot.simulate.cluster.Cluster):
    """Generates the true values of cluster astrometry (not affected by errors)."""
    positions = generate_star_positions(cluster)
    velocities = generate_star_velocities(cluster)

    # Do coordinate frame stuff to get final values (dont look astropy devs, dont tell
    # me what I can and cant do, i live in a lawless realm, this works so it works)
    cluster_center = cluster.parameters.get_position_as_skycoord(
        frame="galactocentric", with_zeroed_proper_motions=True
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
    cluster.cluster["pmra"] = np.nan  # Gets set later
    cluster.cluster["pmdec"] = np.nan  # Gets set later
    cluster.cluster["pmra_true"] = final_coords.pm_ra_cosdec.value
    cluster.cluster["pmdec_true"] = final_coords.pm_dec.value
    cluster.cluster["radial_velocity_true"] = final_coords.radial_velocity.value
    cluster.cluster["parallax_true"] = 1000 / final_coords.distance.value


def generate_cluster_astrometry(cluster: ocelot.simulate.cluster.Cluster):
    """Generates the astrometry of clusters."""
    generate_true_star_astrometry(cluster)
    apply_gaia_astrometric_uncertainties(cluster)
