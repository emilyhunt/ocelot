"""A set of tests for use with the pytest module, covering ocelot.crossmatch"""

from ocelot import crossmatch
from pathlib import Path
import numpy as np
import pandas as pd

test_data_path = Path(__file__).parent / "test_data"

location_mwsc = test_data_path / "mwsc_ii_catalogue"
location_crossmatch_match_positions = test_data_path / "crossmatch_match_position_only"
location_crossmatch_summary_positions = (
    test_data_path / "crossmatch_summary_position_only"
)
location_crossmatch_match_all = test_data_path / "crossmatch_match_all_axes"
location_crossmatch_summary_all = test_data_path / "crossmatch_summary_all_axes"


def test_catalogue():
    """Tests ocelot.crossmatch.Catalogue, the primary tool for cross-matching open 
    clusters by comparing them against a known literature catalogue with arbitrary 
    matchable parameters.
    """
    # Read in the catalogues
    list_of_10_ocs = [
        "Blanco_1",
        "Stock_10",
        "Kharchenko_1",
        "NGC_2516",
        "Trumpler_19",
        "ASCC_79",
        "Hogg_19",
        "NGC_6475",
        "Ruprecht_172",
        "King_9",
    ]

    data_mwsc = pd.read_csv(location_mwsc)

    # ... and make a perturbed version.
    # All clusters have been lightly perturbed in position space and proper motion 
    # space. However, cluster 5 (ASCC 79) has been offset in RA by 2 degrees, and 
    # cluster 9 (King 9) has been offset by 4kpc, making both of these clusters
    # into outliers.
    data_perturbed = (
        data_mwsc[data_mwsc["Name"].isin(list_of_10_ocs)]
        .sort_values("MWSC")
        .reset_index()
        .copy()
    )

    np.random.seed(42)
    data_perturbed[["RAJ2000", "DEJ2000"]] += np.random.normal(
        loc=0.0, scale=0.001, size=(10, 2)
    )
    data_perturbed[["pmRA", "pmDE"]] += np.random.normal(
        loc=0.0, scale=0.5, size=(10, 2)
    )
    data_perturbed["d"] += (
        np.random.normal(loc=0.0, scale=0.1, size=10) * data_perturbed["d"]
    )
    data_perturbed.loc[5, "RAJ2000"] += 2
    data_perturbed.loc[9, "d"] -= 4000
    data_perturbed["ra_error"] = 0.001
    data_perturbed["dec_error"] = 0.001
    data_perturbed["distance_error"] = data_perturbed["d"] * 0.1

    key_override_dict = {
        "name": "Name",
        "ra": "RAJ2000",
        "dec": "DEJ2000",
        "distance": "d",
        "pmra": "pmRA",
        "pmra_error": "e_pm",
        "pmdec": "pmDE",
        "pmdec_error": "e_pm",
    }

    # Create a catalogue with data_mwsc and try to crossmatch against it, but without 
    # any extra axes
    catalogue = crossmatch.Catalogue(
        data_mwsc,
        "mwsc",
        key_name="Name",
        key_ra="RAJ2000",
        key_dec="DEJ2000",
    )

    catalogue.override_default_ocelot_parameter_names(key_override_dict)

    matches, summary = catalogue.crossmatch(data_perturbed, matches_to_record=2)

    # Test that both of the DataFrames for position only matching are correct (they were
    # checked by hand originally)
    pd.testing.assert_frame_equal(
        matches, pd.read_csv(location_crossmatch_match_positions, index_col=0)
    )
    pd.testing.assert_frame_equal(
        summary, pd.read_csv(location_crossmatch_summary_positions, index_col=0)
    )

    # Create a catalogue with data_mwsc and try to crossmatch against it, now with many
    # extra axes
    catalogue = crossmatch.Catalogue(
        data_mwsc,
        "mwsc",
        key_name="Name",
        key_ra="RAJ2000",
        key_dec="DEJ2000",
        extra_axes=[["d", None], ["pmRA", "e_pm"], ["pmDE", "e_pm"]],
    )

    catalogue.override_default_ocelot_parameter_names(key_override_dict)

    matches_all, summary_all = catalogue.crossmatch(
        data_perturbed,
        extra_axes_keys=["distance", "pmra", "pmdec"],
        matches_to_record=2,
    )

    # Test that both of the DataFrames for all axis matching are correct (they were
    # checked by hand originally)
    pd.testing.assert_frame_equal(
        matches_all, pd.read_csv(location_crossmatch_match_all, index_col=0)
    )
    pd.testing.assert_frame_equal(
        summary_all, pd.read_csv(location_crossmatch_summary_all, index_col=0)
    )
