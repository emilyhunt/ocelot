"""Helpers for working with observation models and simulating a cluster observation."""

from __future__ import annotations
from ocelot.model.observation import (
    BaseObservation,
    CustomAstrometricMethodObservation,
    CustomPhotometricMethodObservation,
)
from ocelot.model.observation.common import (
    calculate_separation,
    apply_astrometric_errors_simple_gaussian,
    apply_photometric_errors_simple_gaussian,
)
from ocelot.util.magnitudes import add_two_magnitudes
import ocelot.simulate.cluster


def apply_extinction_to_photometry(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Extinguishes the photometry for this cluster based on the position of each star."""
    model.calculate_extinction(cluster)
    observation = cluster.observations[model.name]
    for band in model.photometric_band_names:
        observation[band] = (
            observation[f"{band}_true"] + observation[f"extinction_{band}"]
        )


def make_unresolved_stars(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Combines stars that are close to one another into single sources."""
    # Todo improve to be able to consider any stars in a dataset, not just binaries
    # Todo this function is too long and complicated
    if not cluster.features.binary_stars:
        return
    observation = cluster.observations[model.name]

    # Perform indexing fuckery to get our primary & secondary stars
    is_secondary = observation["index_primary"] > -1
    primary_indices = observation.loc[is_secondary, "index_primary"].to_numpy()
    secondary_indices = is_secondary.to_numpy().nonzero()[0]
    primary, secondary = (
        observation.loc[primary_indices],
        observation.loc[secondary_indices],
    )

    # Calculate the probability that they're resolved separately
    separations = calculate_separation(primary, secondary)
    probability_separate = model.calculate_resolving_power(
        primary, secondary, separations
    )

    # Randomly decide which ones need blending
    samples = cluster.random_generator.uniform(low=0.0, high=1.0, size=len(secondary))
    needs_blending = probability_separate < samples
    primary_indices_blend = primary_indices[needs_blending]
    secondary_indices_blend = secondary_indices[needs_blending]

    # Add magnitudes
    # Todo can this be sped up? May be hard as each star comes one after another
    bands = model.photometric_band_names
    observation["unresolved_companions"] = 0
    for primary_index, secondary_index in zip(
        primary_indices_blend, secondary_indices_blend
    ):
        observation.loc[primary_index, bands] = add_two_magnitudes(
            observation.loc[primary_index, bands].to_numpy().astype(float),
            observation.loc[secondary_index, bands].to_numpy().astype(float),
        )
        observation.loc[primary_index, "unresolved_companions"] += 1

    # Drop blended stars
    cluster.observations[model.name] = observation.drop(
        secondary_indices_blend
    ).reset_index(drop=True)


def apply_photometric_errors(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Propagates errors into the cluster's photometry."""
    if not cluster.features.photometric_uncertainties:
        return

    model.calculate_photometric_errors(cluster)

    if isinstance(model, CustomPhotometricMethodObservation):
        model.apply_photometric_errors(cluster)
    else:
        apply_photometric_errors_simple_gaussian(cluster, model)


def apply_astrometric_errors(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Propagates errors into the cluster's astrometry."""
    if not cluster.features.astrometric_uncertainties:
        return
    if not (model.has_parallaxes or model.has_proper_motions):
        return
    
    model.calculate_astrometric_errors(cluster)

    if isinstance(model, CustomAstrometricMethodObservation):
        model.apply_astrometric_errors(cluster)
    else:
        apply_astrometric_errors_simple_gaussian(cluster, model)


def apply_selection_function(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Applies selection functions to an observation."""
    if not cluster.features.selection_effects:
        return

    observation = cluster.observations[model.name]

    # Query all selection functions
    selection_functions = model.get_selection_functions(cluster)
    if len(selection_functions) == 0:
        return
    column_names = [func.query(cluster, model.name) for func in selection_functions]

    # Total selection probability is just the product of all of them (Rix+21)
    observation["selection_probability"] = observation[column_names].prod(axis=1)

    # Sample whether or not we see each star
    samples = cluster.random_generator.uniform(0.0, 1.0, len(observation))
    star_is_visible = observation["selection_probability"] > samples

    # Drop missing stars!
    cluster.observations[model.name] = observation.loc[star_is_visible].reset_index(
        drop=True
    )


def cleanup_observation(
    cluster: ocelot.simulate.cluster.SimulatedCluster, model: BaseObservation
):
    """Simple function that performs a few final cleanup tasks, like removing columns
    that we won't want to have around.
    """
    # Remove index_primary because indexing on primary star will be broken now
    columns_to_remove = ["index_primary"]

    cluster.observations[model.name] = cluster.observations[model.name].drop(
        columns=columns_to_remove
    )
