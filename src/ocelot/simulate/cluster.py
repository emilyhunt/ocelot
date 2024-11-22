"""Definition of a base SimualtedCluster class that handles checking and grouping of 
parameters and data.
"""

import numpy as np
from astropy import units as u
from astropy import constants
from astropy.coordinates import SkyCoord
from ocelot.calculate.profile import king_number_density
import pandas as pd
from scipy.optimize import minimize
from ocelot.simulate.errors import NotEnoughStarsError, CoreRadiusTooLargeError
from ocelot.simulate.photometry import generate_cluster_photometry
from ocelot.simulate.astrometry import generate_cluster_astrometry
from ocelot.simulate.binaries import MoeDiStefanoMultiplicityRelation
from dataclasses import dataclass, asdict, field
import dustmaps.map_base


def calculate_r_50(r_core: int | float, r_tidal: int | float):
    """Calculates r_50 for a given King+62 model.
    
    # Todo: move this
    """
    if r_core >= r_tidal:
        raise CoreRadiusTooLargeError("r_core may not be greater than r_tidal!")
    if r_core < 0:
        raise ValueError("r_core must be positive!")
    if r_tidal < 0:
        raise ValueError("r_tidal must be positive!")
    total_value = king_number_density(r_tidal, r_core, r_tidal)
    target_value = total_value / 2

    def func_to_minimise(r):
        return (target_value - king_number_density(r, r_core, r_tidal)) ** 2

    result = minimize(
        func_to_minimise,
        np.atleast_1d([r_core]),
        method="Nelder-Mead",
        bounds=((0.0, r_tidal),),
    )

    if not result.success:
        raise RuntimeError(
            f"unable to find an r_50 value given r_core={r_core} and r_tidal={r_tidal}"
        )

    return result.x[0]


def calculate_velocity_dispersion_1d(r_50, mass, virial_ratio, eta=10.0):
    """Calculates the 1D velocity dispersion of a cluster given its current virial
    state. See Portegies-Zwart+10 / Emily's thesis for equation derivation

    sigma_1d = [ ( 2 * Q * G * M) / (eta * r_50) ]^0.5

    # Todo: move this
    """
    mass_kg = (mass * u.M_sun).to(u.kg).value
    r_50_m = (r_50 * u.pc).to(u.m).value
    return ((2 * virial_ratio * constants.G.value * mass_kg) / (eta * r_50_m)) ** 0.5


DEFAULT_BAYESTAR_WEB_QUERY = dustmaps.bayestar.BayestarWebQuery()


@dataclass
class SimulatedClusterParameters:
    """Class for keeping track of parameters specified for a cluster to simulate."""
    # Todo: all of these should really be astropy quantity objects. Units!!!
    position: SkyCoord
    mass: float
    log_age: float
    metallicity: float
    extinction: float
    r_core: float
    r_tidal: float
    # profile: object = ...  # Todo: support multiple distribution profiles
    minimum_stars: int = 1
    virial_ratio: float | int = 0.5
    velocity_dispersion_1d: float | None = None
    eta_virial_ratio: float | int = 10.0
    radial_velocity: float | int = 0.0
    photometric_errors: bool = False
    astrometric_errors: bool = True
    astrometric_errors_scale_factor: float | int = 1.0
    selection_effects: bool = True
    visible_stars_only: bool = True
    binary_stars: bool = True
    binary_star_relation: object = MoeDiStefanoMultiplicityRelation
    id: int = 0

    # The following fields are calculated in __post_init__, as they depend on initial
    # values:
    r_50: float = field(init=False)
    n_stars: int = field(init=False, default=0)
    dust_map: str = field(init=False, default="")
    ra: float = field(init=False)
    dec: float = field(init=False)
    l: float = field(init=False)  # noqa: E741
    b: float = field(init=False)
    distance: float = field(init=False)
    pmra: float = field(init=False)
    pmdec: float = field(init=False)
    radial_velocity: float = field(init=False)

    def __post_init__(self):
        """Automatically calculates any additional things and does all checks."""
        self.r_50 = calculate_r_50(self.r_core, self.r_tidal)

        # Velocity-related things
        self.velocity_dispersion_1d = calculate_velocity_dispersion_1d(
            self.r_50, self.mass, self.virial_ratio, self.eta_virial_ratio
        )

        # Ensure that we set positions etc. to this class. This makes them more
        # easily convertible to a dict at a later date.
        position_icrs = self.position.transform_to("icrs")
        position_galactic = self.position.transform_to("galactic")
        self.ra = position_icrs.ra.to(u.deg).value 
        self.dec = position_icrs.dec.to(u.deg).value
        self.distance = position_icrs.distance.to(u.pc).value 
        self.pmra = position_icrs.pm_ra_cosdec.to(u.mas / u.yr).value 
        self.pmdec = position_icrs.pm_dec.to(u.mas / u.yr).value
        self.radial_velocity = position_icrs.radial_velocity.to(u.m / u.s).value
        self.l = position_galactic.l.to(u.deg).value
        self.b = position_galactic.b.to(u.deg).value 

        self.check()

    def check(self):
        """Checks that the cluster has parameters that are correctly specified."""
        if self.r_core >= self.r_tidal:
            raise CoreRadiusTooLargeError(
                "specified core radius larger than calculated tidal radius!"
            )

    def get_position_as_skycoord(self, frame="icrs", with_zeroed_proper_motions=False):
        """Returns the position of the cluster as a SkyCoord."""
        kwargs = dict(radial_velocity=self.radial_velocity * u.km / u.s)
        if with_zeroed_proper_motions:
            kwargs["pm_l_cosb"] = 0 * u.mas / u.yr
            kwargs["pm_b"] = 0 * u.mas / u.yr

        return SkyCoord(
            l=self.l * u.deg,
            b=self.b * u.deg,
            distance=self.distance * u.pc,
            frame="galactic",
            **kwargs,
        ).transform_to(frame)

    def to_dict(self):
        """Converts class (and all fields) to dict. Useful for saving information."""
        return asdict(self)


class SimulatedCluster:
    def __init__(
        self,
        random_seed: int,
        parameters: SimulatedClusterParameters | None,
        **kwargs
    ):
        """This is a helper class used to specify the parameters of a cluster to
        simulate.
        """
        if parameters is None:
            parameters = SimulatedClusterParameters(**kwargs)
        self.parameters: SimulatedClusterParameters = parameters

        # Stuff for handling randomness
        # Todo: IMF package isn't using these and there's seemingly no way to fix it
        self.random_seed = random_seed
        self.random_generator = np.random.default_rng(random_seed)

        # Empty things
        self.cluster: pd.DataFrame = (
            pd.DataFrame()
        )  # Initialise blank to shut pylance up
        self.stars: int = 0
        self.photometry_made: bool = False
        self.astrometry_made: bool = False

    def _check_if_has_enough_stars(self):
        """Checks if the cluster has enough stars."""
        if self.stars < self.parameters.minimum_stars:
            raise NotEnoughStarsError(
                f"Simulated cluster only contains {self.stars} stars, which is less "
                f"than the minimum value of {self.parameters.minimum_stars} that was "
                "specified!"
            )

    def make_photometry(self, field: None | pd.DataFrame = None):
        """Generates photometry for this cluster given its own parameters."""
        if self.photometry_made:
            raise RuntimeError(
                "Photometry for this cluster was already made! Cannot run again."
            )
        generate_cluster_photometry(self, field)
        self._check_if_has_enough_stars()
        self.photometry_made = True

    def make_astrometry(self):
        """Generates astrometry for this cluster. Only works if the cluster has
        photometry already.
        """
        if self.astrometry_made:
            raise RuntimeError(
                "Astrometry for this cluster was already made! Cannot run again."
            )
        if not self.photometry_made:
            raise RuntimeError(
                "Photometry is required to generate astrometry! Try doing "
                "make_photometry first."
            )
        generate_cluster_astrometry(self)
        self.astrometry_made = True

        # It's assumed that users will usually want to set proper motions many times
        # after generating the cluster, but we also support setting it in the initial
        # parameter object. Calling this initially will by default just set it to (0,0).
        self.set_proper_motion(self.parameters.pmra, self.parameters.pmdec)

    def make(self, field: None | pd.DataFrame = None):
        """Generates photometry and astrometry for the cluster."""
        self.make_photometry(field)
        self.make_astrometry()

    # def plot(self, field: pd.DataFrame | None = None, fig=None, ax=None, **kwargs):
    #     """Plots the current cluster using oc_selection.plots.cluster_plot.

    #     Parameters
    #     ----------
    #     field : pd.DataFrame | None, optional
    #         _description_, by default None
    #     fig : _type_, optional
    #         _description_, by default None
    #     ax : _type_, optional
    #         _description_, by default None
    #     kwargs : dict, optional
    #         Additional keyword arguments to pass to oc_selection.plots.cluster_plot

    #     Returns
    #     -------
    #     _type_
    #         _description_

    #     Raises
    #     ------
    #     ImportError
    #         If oc_selection is not installed.
    #     """
    #     # Import here as this is an optional dependency
    #     try:
    #         from oc_selection.plots import cluster_plot
    #     except ImportError:
    #         raise ImportError("oc_selection library not found! Unable to plot cluster.")

    #     return cluster_plot([self], field, fig, ax, **kwargs)
