"""This is a script to download data used for testing ocelot. The data itself lives on
GitHub - this script exists to reproduce it.
"""

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from pathlib import Path

test_data_dir = Path("../tests/test_data")
if not test_data_dir.exists():
    raise ValueError(
        "Unable to find test data directory. Are you running this script in scripts/?"
    )


# Download a small patch of Gaia data for testing
coordinates = SkyCoord(45, 0, unit="deg")
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = 5000
query = Gaia.cone_search(coordinates, radius=0.5 * u.deg)
gaia_data = query.get_results().to_pandas().rename(columns={"SOURCE_ID": "source_id"})
gaia_data.to_parquet(test_data_dir / "gaia/dr3/ra=45,dec=0,radius=0.5.parquet")


