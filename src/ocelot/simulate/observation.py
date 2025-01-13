"""Helpers for working with observation models and simulating a cluster observation."""

from __future__ import annotations
from ocelot.model.observation import BaseObservation
from ocelot.simulate import SimulatedCluster
from ocelot.util.magnitudes import add_two_magnitudes


def apply_extinction_to_photometry(cluster: SimulatedCluster, model: BaseObservation):
    """Extinguishes the photometry for this cluster based on the position of each star."""
    model.calculate_extinction(cluster)
    observation = cluster.observations[model.name]
    for band in model.photometric_band_names:
        observation[band] = (
            observation[f"{band}_true"] + observation[f"extinction_{band}"]
        )


def make_unresolved_stars(cluster: SimulatedCluster, model: BaseObservation):
    """Combines stars that are close to one another into single sources."""
    # Todo improve to be able to consider any stars in a dataset, not just binaries
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
    probability_separate = model.calculate_resolving_power(primary, secondary)
    samples = cluster.random_generator.uniform(low=0.0, high=1.0, size=len(secondary))
    needs_blending = probability_separate < samples
    primary_indices_blend = primary_indices[needs_blending]
    secondary_indices_blend = secondary_indices[needs_blending]

    # Add magnitudes
    # Todo can this be sped up? May be hard as each star comes one after another
    bands = model.photometric_band_names
    for primary_index, secondary_index in zip(
        primary_indices_blend, secondary_indices_blend
    ):
        observation.loc[primary_index, bands] = add_two_magnitudes(
            observation.loc[primary_index, bands].to_numpy(),
            observation.loc[secondary_index, bands].to_numpy(),
        )

    # Drop blended stars
    observation = observation.drop(secondary_indices_blend).reset_index(drop=True)


def apply_errors(cluster: SimulatedCluster, model: BaseObservation):
    """Propagates errors into the cluster's photometry and astrometry."""
    observation = cluster.observations[model.name]
    model.calculate_photometric_errors(cluster)
    for band in model.photometric_band_names:
        new_fluxes = cluster.random_generator.normal(
            loc=model.mag_to_flux(observation[band].to_numpy(), band),
            scale=observation[f"{band}_flux_error"].to_numpy(),
        )
        observation[band] = model.flux_to_mag(new_fluxes, band)

    astrometric_columns_with_error = []
    if model.has_parallaxes:
        astrometric_columns_with_error.append("parallax")
    if model.has_proper_motions:
        astrometric_columns_with_error.extend(["pmra", "pmdec"])
    if len(astrometric_columns_with_error) == 0:
        return

    model.calculate_astrometric_errors(cluster)
    for column in astrometric_columns_with_error:
        observation[column] = cluster.random_generator.normal(
            loc=observation[column].to_numpy(),
            scale=observation[f"{column}_error"].to_numpy(),
        )


def apply_selection_function(cluster: SimulatedCluster, model: BaseObservation):
    """Applies selection functions to an observation."""
    observation = cluster.observations[model.name]

    # Query all selection functions
    selection_functions = model.get_selection_functions()
    if len(selection_functions) == 0:
        return
    column_names = [func(cluster, model.name) for func in selection_functions]

    # Total selection probability is just the product of all of them (Rix+21)
    observation["selection_probability"] = observation[column_names].prod(axis=1)

    # Sample whether or not we see each star
    samples = cluster.random_generator.uniform(0.0, 1.0, len(observation))
    star_is_visible = observation["selection_probability"] > samples
    cluster.cluster.loc[star_is_visible].reset_index(drop=True)
