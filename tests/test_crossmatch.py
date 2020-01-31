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
import pytest

location_mwsc = Path("./test_data/mwsc_ii_catalogue")


def test_catalogue():
    """Tests ocelot.crossmatch.Catalogue, the primary tool for cross-matching open clusters by comparing them against
    a known literature catalogue with arbitrary matchable parameters."""
    # Read in the MWSC catalogue and create a test catalogue based loosely on it



