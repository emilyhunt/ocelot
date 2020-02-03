"""A set of tests for use with the pytest module, covering ocelot.crossmatch"""

# FUCKING HATE PYTHON IMPORTS AAAA
# (the below fixes this though)
try:
    from .context import ocelot
except ModuleNotFoundError:
    print('Unable to find ocelot via .context! Trying to import from your python path instead...')
try:
    import ocelot
except ModuleNotFoundError:
    raise ModuleNotFoundError('Unable to find ocelot')

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

location_mwsc = Path("./test_data/mwsc_ii_catalogue")
location_mwsc_perturbed = Path("./test_data/mwsc_ii_perturbed")


def test_catalogue():
    """Tests ocelot.crossmatch.Catalogue, the primary tool for cross-matching open clusters by comparing them against
    a known literature catalogue with arbitrary matchable parameters.

    -------------------

    The test catalogue was made from the MWSC one, with:

    data_cut = data_mwsc[data_mwsc["Name"].isin(list_of_10_ocs)].sort_values("MWSC").reset_index()
    data_fiddled = data_cut.copy()

    np.random.seed(42)
    data_fiddled[["RAJ2000", "DEJ2000"]] += np.random.normal(loc=0.0, scale=0.25, size=(10, 2))
    data_fiddled[["pmRA", "pmDE"]] += np.random.normal(loc=0.0, scale=0.5, size=(10, 2))
    data_fiddled["d"] += np.random.normal(loc=0.0, scale=0.1, size=10) * data_fiddled["d"]
    data_fiddled.loc[5, "RAJ2000"] += 2
    data_fiddled.loc[9, "d"] -= 4000
    data_fiddled["e_d"] = data_fiddled["d"] * 0.1

    data_fiddled.to_csv("mwsc_ii_perturbed")

    -------------------

    All clusters have been lightly perturbed in position space and proper motion space. However, cluster 5 (ASCC 79)
    has been offset in RA by 2 degrees, and cluster 9 (King 9) has been offset by 4kpc, making both of these clusters
    into outliers.

    """
    # Read in the catalogues
    data_mwsc = pd.read_csv(location_mwsc)
    data_perturbed = pd.read_csv(location_mwsc_perturbed)

    # Create a catalogue with data_mwsc and try to crossmatch against it, but without any extra axes
    catalogue = ocelot.crossmatch.Catalogue(data_mwsc,
                                            "mwsc",
                                            key_name="Name",
                                            key_ra="RAJ2000",
                                            key_dec="DEJ2000",)

    matches, summary = catalogue.crossmatch(
        data_perturbed["Name"],
        data_perturbed["RAJ2000"],
        data_perturbed["DEJ2000"],
        data_perturbed["r0"],
        matches_to_record=5)

    return matches, summary, data_mwsc, data_perturbed


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    match, summ, mwsc, pert = test_catalogue()


