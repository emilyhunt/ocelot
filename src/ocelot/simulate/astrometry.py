"""Functions for dealing with astrometry-related things."""

from __future__ import annotations  # Necessary to type hint without cyclic import
import numpy as np
from ocelot.calculate.profile import sample_1d_king_profile
from ocelot.util.random import points_on_sphere
import ocelot.simulate.cluster
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from scipy.stats import multivariate_normal


def generate_star_positions(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Generates positions of member stars in polar coordinates relative to the center
    of the cluster.
    """
    x, y, z = cluster.models.distribution.rvs(
        len(cluster.cluster), seed=cluster.random_generator
    ).T

    # Also handle binary star positions
    if cluster.parameters.binary_stars:
        x, y, z = generate_binary_star_positions(cluster, x, y, z)
    return CartesianRepresentation(x, y, z, unit=u.pc)


def generate_binary_star_positions(
    cluster: ocelot.simulate.cluster.SimulatedCluster, x, y, z
):
    """Generates locations relative to their host star for secondaries in the cluster."""

    pass


def generate_star_velocities(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Generates the velocities of stars in a cluster."""
    # Todo should get velocities from the distribution object eventually
    distribution = multivariate_normal(
        mean=np.zeros(3),
        cov=np.identity(3) * cluster.parameters.velocity_dispersion_1d,
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

    # Todo binary star positions should be offset from their host star

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
