"""Binary star relations for clusters."""

from __future__ import annotations
import numpy as np
import pickle
import ocelot.simulate.cluster
from scipy.interpolate import interp1d
from numba import jit
from ocelot import DATA_PATH


# The below values are hard-coded directly from the tables of the following papers.
# Stellar masses 1+ Msun: Moe & DiStefano 2017
# Below this: Duchene & Kraus 2013
mass = np.asarray([0.10, 0.30, 1.00, 3.50, 7.00, 12.5, 16.0])
multiplicity_fraction = np.asarray([0.22, 0.26, 0.40, 0.59, 0.76, 0.84, 0.94])
companion_star_frequency = np.asarray([0.22, 0.33, 0.50, 0.84, 1.30, 1.60, 2.10])

MF_INTERPOLATOR = interp1d(
    mass,
    multiplicity_fraction,
    bounds_error=False,
    fill_value=(multiplicity_fraction.min(), multiplicity_fraction.max()),
)

CSF_INTERPOLATOR = interp1d(
    mass,
    companion_star_frequency,
    bounds_error=False,
    fill_value=(companion_star_frequency.min(), companion_star_frequency.max()),
)


location_random_q_interpolator = (
    DATA_PATH / "binaries/Moe_DiStefano17_interpolated_random_q_relation.pickle"
)
with open(location_random_q_interpolator, "rb") as handle:
    MOE_DI_STEFANO_RANDOM_Q_INTERPOLATOR = pickle.load(handle)


class MoeDiStefanoMultiplicityRelation:
    def __init__(
        self,
        make_resolved_stars=True,
        distance=None,
        separation=0.6,
        interpolated_q=True,
    ) -> None:
        """An interpolated implementation of the MoeDiStefano17 multiplicity relations,
        plus DucheneKraus+13 for stars below 1 MSun.
        """
        self.make_resolved_stars = make_resolved_stars
        self.distance = distance
        self.separation = separation

        if not interpolated_q:
            raise ValueError(
                "interpolated_q may only be set to True. This implementation only "
                "supports using pre-computed data from Hunt & Reffert 2024."
            )

        with open(location_random_q_interpolator, "rb") as handle:
            self.interpolator = pickle.load(handle)

        # If no distance specified in interpolation mode, then set to max distance
        # (where everything is unresolved anyway)
        if self.distance is None or make_resolved_stars is False:
            self.distance = np.max(self.interpolator.grid[1])

    def companion_star_fraction(self, masses):
        return CSF_INTERPOLATOR(masses)

    def multiplicity_fraction(self, masses):
        return MF_INTERPOLATOR(masses)

    def random_q(self, masses, random_generator):
        masses = np.atleast_1d(masses)
        distances = np.repeat(self.distance, len(masses))
        samples = random_generator.uniform(size=len(masses))
        points = np.vstack([masses, distances, samples]).T
        return self.interpolator(points)


def _get_multiplicity_relations(cluster):
    multiplicity_relation_all_binaries = cluster.parameters.binary_star_relation(
        make_resolved_stars=False
    )
    multiplicity_relation_resolved_binaries = cluster.parameters.binary_star_relation(
        make_resolved_stars=True, distance=cluster.parameters.distance
    )
    return multiplicity_relation_all_binaries, multiplicity_relation_resolved_binaries


def _assign_number_of_companions(cluster, multiplicity_relation_all_binaries):
    """Assigns a number of companions to each star probabilistically."""
    expected_companions = multiplicity_relation_all_binaries.companion_star_fraction(
        cluster.cluster["mass"]
    )
    expected_multiplicity = multiplicity_relation_all_binaries.multiplicity_fraction(
        cluster.cluster["mass"]
    )
    n_stars = len(cluster.cluster)
    is_multiple = cluster.random_generator.uniform(size=n_stars) < expected_multiplicity
    total_multiples = is_multiple.sum()

    cluster.cluster["companions"] = np.zeros(n_stars, dtype=int)
    cluster.cluster.loc[is_multiple, "companions"] = (
        np.random.poisson(
            (expected_companions[is_multiple] / expected_multiplicity[is_multiple]) - 1,
            size=total_multiples,
        )
        + 1
    ).astype(int)


@jit(nopython=True, cache=True)
def _convert_singles_numba(
    masses, companions, indices_to_go_over, index_primary, q_values, is_resolved
):
    """Fast inner loop for convert_singles_to_systems.

    It's very weirdly written so that it works with numba (I had to really wrestle with
    making it optimisation-friendly), sorry :D

    # Todo: make this function non-shit
    """
    index_into_companions = 0
    all_star_indices = np.arange(masses.size)
    valid_masses = np.ones(masses.size, dtype=np.bool_)

    for i_primary, n_companions in zip(indices_to_go_over, companions):
        # -------------
        # CHECKS
        # -------------
        # Skip stars that are already binaries
        if index_primary[i_primary] > -2:
            index_into_companions += n_companions
            continue
        # Can't run on the lowest-mass star
        if i_primary == 0:
            continue

        # -------------
        # GET DATA FROM OVERALL ARRAYS INTO SHORTER ONES
        # -------------
        primary_mass = masses[i_primary]
        valid_masses[i_primary] = False

        # Stop if we've ran out of valid potential companions
        if np.sum(valid_masses) == 0:
            return index_primary

        mass_companions = (
            q_values[index_into_companions : index_into_companions + n_companions]
            * primary_mass
        )
        resolved_companions = is_resolved[
            index_into_companions : index_into_companions + n_companions
        ]

        # Increment for next loop
        index_into_companions += n_companions

        # -------------
        # CYCLE OVER EVERY COMPANION AND SELECT
        # -------------
        # Get (unique) indices of companions
        for a_mass, a_resolved in zip(mass_companions, resolved_companions):
            # Find star closest in mass to the one required by q
            i_secondary = np.searchsorted(masses[valid_masses], a_mass)

            # Get a valid index for this star
            indices_to_search = all_star_indices[valid_masses]

            # Prevent issues if the best star is at the end of the array
            if i_secondary >= len(indices_to_search):
                i_secondary = len(indices_to_search) - 1

            # Now, finally, we can grab the index of the star and set that it's now a
            # secondary!
            i_secondary_converted = indices_to_search[i_secondary]
            valid_masses[i_secondary_converted] = False

            # If the star is resolved (randomly sampled for now), then assign it as
            # such; otherwise, set it to contribute to the primary star's magnitude
            # (i.e. be unresolved), meaning it later gets removed from the star list
            if a_resolved:
                index_primary[i_secondary_converted] = -1
            else:
                index_primary[i_secondary_converted] = i_primary

            # Stop if we've ran out of valid potential companions
            if np.sum(valid_masses) == 0:
                return index_primary

    return index_primary


def _convert_singles_to_systems(
    cluster, multiplicity_relation_all_binaries, multiplicity_relation_resolved_binaries
):
    """Converts single stars to systems, based on precomputed appropriate numbers of
    companions to make for each one.

    # Todo tidy this function please it's a mess
    """
    query = "companions > 0"
    if cluster.parameters.selection_effects:
        query += " and g_true < 21"
    indices_to_go_over = cluster.cluster.query(query).index.to_numpy()[::-1]

    masses = cluster.cluster["mass"].to_numpy()
    companions = cluster.cluster.loc[indices_to_go_over, "companions"].to_numpy()
    index_primary = np.full(len(cluster.cluster), -2)

    masses_repeated = np.repeat(masses[indices_to_go_over], companions)
    q_values = multiplicity_relation_all_binaries.random_q(
        masses_repeated, cluster.random_generator
    )

    # Todo: Whether or not stars are resolved is currently done independent of mass. This is wrong! Minimal impact on distant clusters but bad in other cases.
    q_resolved = multiplicity_relation_resolved_binaries.random_q(
        masses_repeated, cluster.random_generator
    )
    is_resolved = q_resolved < 0.1  # Weird thing with how the interp is specced

    cluster.cluster["index_primary"] = _convert_singles_numba(
        masses, companions, indices_to_go_over, index_primary, q_values, is_resolved
    )

    # Add fluxes
    summed_fluxes_of_companions = (
        cluster.cluster.query("index_primary >= 0")
        .groupby("index_primary")[["g_flux", "bp_flux", "rp_flux"]]
        .sum()
    )

    primary_indices = summed_fluxes_of_companions.index
    cluster.cluster.loc[primary_indices, "g_flux"] += summed_fluxes_of_companions[
        "g_flux"
    ]
    cluster.cluster.loc[primary_indices, "bp_flux"] += summed_fluxes_of_companions[
        "bp_flux"
    ]
    cluster.cluster.loc[primary_indices, "rp_flux"] += summed_fluxes_of_companions[
        "rp_flux"
    ]

    # Drop the taken stars
    cluster.cluster = cluster.cluster.query("index_primary < 0").reset_index(drop=True)


# Zeropoints in the Vegamag system (see documentation table 5.2)
# These are for Gaia DR3!
G_ZP = 25.6874
BP_ZP = 25.3385
RP_ZP = 24.7479


def _mag_to_flux(magnitudes, zero_point):
    return 10 ** ((zero_point - magnitudes) / 2.5)


def _flux_to_mag(fluxes, zero_point):
    # We also handle negative fluxes here - in that case, it should just be inf
    good_fluxes = np.atleast_1d(fluxes > 0).flatten()
    magnitudes = (
        -2.5 * np.log10(np.atleast_1d(fluxes).flatten(), where=good_fluxes) + zero_point
    )
    magnitudes[np.invert(good_fluxes)] = np.inf
    return magnitudes


def _calculate_fluxes(cluster):
    """Calculate fluxes for stars in the cluster"""
    good_stars = cluster.cluster["g_true"].notna()
    bad_stars = np.invert(good_stars)
    cluster.cluster.loc[good_stars, "g_flux"] = _mag_to_flux(
        cluster.cluster.loc[good_stars, "g_true"], G_ZP
    )
    cluster.cluster.loc[good_stars, "bp_flux"] = _mag_to_flux(
        cluster.cluster.loc[good_stars, "bp_true"], BP_ZP
    )
    cluster.cluster.loc[good_stars, "rp_flux"] = _mag_to_flux(
        cluster.cluster.loc[good_stars, "rp_true"], RP_ZP
    )
    cluster.cluster.loc[bad_stars, "g_flux"] = 0.0
    cluster.cluster.loc[bad_stars, "bp_flux"] = 0.0
    cluster.cluster.loc[bad_stars, "rp_flux"] = 0.0


def _recalculate_magnitudes(cluster):
    """Recalculate magnitudes for stars in the cluster, after adding binaries."""
    good_stars = cluster.cluster["g_flux"] > 0
    bad_stars = np.invert(good_stars)
    cluster.cluster.loc[good_stars, "g_true"] = _flux_to_mag(
        cluster.cluster.loc[good_stars, "g_flux"], G_ZP
    )
    cluster.cluster.loc[good_stars, "bp_true"] = _flux_to_mag(
        cluster.cluster.loc[good_stars, "bp_flux"], BP_ZP
    )
    cluster.cluster.loc[good_stars, "rp_true"] = _flux_to_mag(
        cluster.cluster.loc[good_stars, "rp_flux"], RP_ZP
    )
    cluster.cluster.loc[bad_stars, "g_true"] = np.nan
    cluster.cluster.loc[bad_stars, "bp_true"] = np.nan
    cluster.cluster.loc[bad_stars, "rp_true"] = np.nan
    # cluster.cluster = cluster.cluster.drop(columns=["g_flux", "bp_flux", "rp_flux"])


def make_binaries(cluster: ocelot.simulate.cluster.SimulatedCluster):
    """Pairs a simulated cluster up into binaries."""
    if not cluster.parameters.binary_stars:
        return

    multiplicity_relation_all, multiplicity_relation_resolved = (
        _get_multiplicity_relations(cluster)
    )

    _assign_number_of_companions(cluster, multiplicity_relation_all)

    # No need to go any further if nobody is going to be a binary
    if cluster.cluster["companions"].sum() == 0:
        return

    # Do some setup of the dataframe
    cluster.cluster = cluster.cluster.sort_values("mass", ignore_index=True)
    _calculate_fluxes(cluster)

    # Cycle over every star, giving it companions
    _convert_singles_to_systems(
        cluster,
        multiplicity_relation_all,
        multiplicity_relation_resolved,
    )

    _recalculate_magnitudes(cluster)
