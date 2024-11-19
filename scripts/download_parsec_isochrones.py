"""Script to download isochrones from PARSEC with ezpadova. Intended for development
use only. Downloads to ../data/.... .
"""

import ezpadova
import pandas as pd
import numpy as np
import time
from pathlib import Path


outdir = Path("../data/isochrones/PARSEC_v1.2S")
outdir.mkdir(exist_ok=True, parents=True)

minimum_age = 5.0
break_point = 8.0
maximum_age = 10.15
age_spacing = 0.01

minimum_metallicity = -2.2
maximum_metallicity = 0.5
metallicity_spacing = 0.05

photometric_system = "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat"


metallicities = np.linspace(
    minimum_metallicity,
    maximum_metallicity,
    num=int((maximum_metallicity - minimum_metallicity) / metallicity_spacing + 1),
)
for a_metallicity in metallicities:
    print(f"Working on [M/H]={a_metallicity:.3f}")
    # Download in two batches to get around size limits
    isochrones_first_half = ezpadova.get_isochrones(
        logage=(minimum_age, break_point, age_spacing),
        MH=(a_metallicity, a_metallicity, 0.0),
        photsys_file=photometric_system,
        # track_parsec="parsec_CAF09_v1.2S",
        # track_colibri="parsec_CAF09_v1.2S_NOV13",
    )
    isochrones_second_half = ezpadova.get_isochrones(
        logage=(break_point + age_spacing, maximum_age, age_spacing),
        MH=(a_metallicity, a_metallicity, 0.0),
        photsys_file=photometric_system,
        # track_parsec="parsec_CAF09_v1.2S",
        # track_colibri="parsec_CAF09_v1.2S_NOV13",
    )

    isochrones = pd.concat(
        [isochrones_first_half, isochrones_second_half], ignore_index=True
    )
    isochrones.to_parquet(outdir / f"mh={a_metallicity:.3f}.parquet")

    print("Sleeping to avoid rate limit...")
    time.sleep(5)
