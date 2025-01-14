"""Definition of a base SimualtedCluster class that handles checking and grouping of
parameters and data.
"""

from __future__ import annotations
import numpy as np
from astropy import units as u
from astropy import constants
from astropy.coordinates import SkyCoord
import pandas as pd
from ocelot.simulate.errors import NotEnoughStarsError, CoreRadiusTooLargeError
from ocelot.simulate.photometry import create_population, apply_extinction
from ocelot.simulate.astrometry import generate_true_star_astrometry
from ocelot.simulate.binaries import make_binaries
from ocelot.simulate.observation import (
    apply_extinction_to_photometry,
    make_unresolved_stars,
    apply_errors,
    apply_selection_function,
)
from ocelot.model.binaries import BaseBinaryStarModel, MoeDiStefanoMultiplicityRelation
from ocelot.model.differential_reddening import (
    BaseDifferentialReddeningModel,
    FractalDifferentialReddening,
)
from ocelot.model.distribution import (
    BaseClusterDistributionModel,
    King62,
    Implements3DMethods,
)
from ocelot.model.observation import BaseObservation
from dataclasses import dataclass, asdict, field


SUPPORTED_OBSERVATIONS = ["gaia_dr3"]


@dataclass
class SimulatedClusterParameters:
    """Class for keeping track of parameters specified for a cluster to simulate."""

    # Todo: all of these should really be astropy quantity objects. Units!!!
    position: SkyCoord
    mass: float
    log_age: float
    metallicity: float
    r_core: float  # Todo these should be optional, so that e.g. a Plummer profile can be supported
    r_tidal: float  # Todo these should be optional, so that e.g. a Plummer profile can be supported
    extinction: float
    differential_extinction: float = 0.0
    minimum_stars: int = 1
    virial_ratio: float | int = 0.5
    velocity_dispersion_1d: float | None = None
    eta_virial_ratio: float | int = 10.0
    photometric_errors: bool = True
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
        self.r_50 = (
            King62(self.r_core * u.pc, self.r_tidal * u.pc).r_50.value
        )  # Todo change where r_50 comes from. Probably need some fancy referencing etc

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


@dataclass
class SimulatedClusterModels:
    """Class for keeping track of all models that a generated SimulatedCluster will use."""

    distribution: BaseClusterDistributionModel | None = None
    binaries: BaseBinaryStarModel | None = None
    differential_reddening: BaseDifferentialReddeningModel | None = None
    observations: list[BaseObservation] | tuple[BaseObservation] = tuple()
    observations_dict: dict[str, BaseObservation] = field(init=False)

    def __post_init__(self):
        if self.distribution is not None and not isinstance(
            self.distribution, Implements3DMethods
        ):
            raise ValueError(
                "Specified distribution must implement 3D methods, i.e. it must "
                "subclass Implements3DMethods."
            )
        self.observations_dict = {
            observation.name: observation for observation in self.observations
        }

    def initialise_defaults(self, parameters: SimulatedClusterParameters, seed: int):
        """For all class attributes, replace None values with sensible default models.

        Parameters
        ----------
        seed : int
            Random seed to use for default models that incorporate randomness.
        """
        if self.distribution is None:
            self.distribution = King62(
                parameters.r_core * u.pc,
                parameters.r_tidal * u.pc,  # Todo should have units by default
            )
        if self.binaries is None:
            self.binaries = MoeDiStefanoMultiplicityRelation()
        if self.differential_reddening is None:
            self.differential_reddening = FractalDifferentialReddening(seed=seed)
        return self

    @staticmethod
    def with_default_options(parameters: SimulatedClusterParameters, seed: int):
        return SimulatedClusterModels().initialise_defaults(
            parameters=parameters, seed=seed
        )


class SimulatedCluster:
    def __init__(
        self,
        parameters: SimulatedClusterParameters | dict,
        observations: list[str]
        | None = None,  # Todo consider removing - we can just use the models list
        models: SimulatedClusterModels | dict | None = None,
        prune_simulated_cluster: dict | None = None,  # Todo also make it do something
        random_seed: int | None = None,
    ):
        """This is a helper class used to specify the parameters of a cluster to
        simulate.
        """
        # Todo fstring docs
        # Stuff for handling randomness
        if random_seed is None:
            # Select a random seed from 0 to the largest possible signed 64 bit int
            # This enforces seeding even when a user doesn't specify a seed - helping
            # to ensure reproducibility.
            random_seed = np.random.default_rng().integers(2**63 - 1)

        # Set various state things
        self.random_seed: int = random_seed
        self.random_generator: np.random.Generator = np.random.default_rng(random_seed)

        # Set legacy numpy seed for IMF package's benefit
        # Todo: IMF package isn't using these and there's seemingly no way to fix it without a PR
        np.random.seed(random_seed)

        # Handle parameter input
        if isinstance(parameters, dict):
            parameters = SimulatedClusterParameters(**parameters)
        self.parameters: SimulatedClusterParameters = parameters

        # Handle model input
        if models is None:
            models = dict()
        if isinstance(models, dict):
            models = SimulatedClusterModels(**models)
        models.initialise_defaults(parameters, self.random_seed)
        self.models = models

        # Handle observation input
        if observations is None:
            observations = []
            if len(self.models.observations_dict) > 0:
                observations = list(self.models.observations_dict.keys())
        self._observations_to_make: list[str] = observations

        # Empty things
        self.isochrone: pd.DataFrame = pd.DataFrame()
        self.cluster: pd.DataFrame = pd.DataFrame()
        self.observations: dict[str, pd.DataFrame] = {}
        self.stars: int = 0
        self._true_cluster_generated: bool = False
        self._observations_generated: bool = False

    def _check_if_has_enough_stars(self):
        """Checks if the cluster has enough stars."""
        if self.stars < self.parameters.minimum_stars:
            raise NotEnoughStarsError(
                f"Simulated cluster only contains {self.stars} stars, which is less "
                f"than the minimum value of {self.parameters.minimum_stars} that was "
                "specified!"
            )

    def _reseed_random_generator(self, seed: int):
        self.random_seed = seed
        self.random_generator = np.random.default_rng(seed)
        np.random.seed(seed)

    def make(self):
        """Makes entire cluster according to specification set at initialization."""
        self.make_cluster()
        self.make_observations()

    def make_cluster(self):
        """Creates the true stars and positions in a cluster."""
        if self._true_cluster_generated:
            raise RuntimeError(
                "Cluster already made! You already called make_cluster or make once, "
                "and cannot do so again."
            )
        create_population(self)
        make_binaries(self)
        generate_true_star_astrometry(self)
        apply_extinction(self)
        self._true_cluster_generated = True
        return self.cluster

    def make_observations(self):
        """Makes all observations of the cluster."""
        for observation in self._observations_to_make:
            self.make_observation(observation)
        self._observations_generated = True

    def make_observation(self, survey: str, seed=None):
        """Makes one observation of the cluster."""
        if not self._true_cluster_generated:
            raise RuntimeError(
                "You must make the true cluster first before generating observations, "
                "either by calling make_cluster or make."
            )
        # Allow for per-observation seed: useful when making many observations of the
        # same cluster.
        if seed is not None:
            self._reseed_random_generator(seed)

        # Fetch the model we need
        if survey not in self.models.observations_dict:
            raise ValueError(
                f"You must specify an observation model for survey {survey} to generate"
                " an observation of it."
            )
        model = self.models.observations_dict[survey]

        # Create the observation!
        self.observations[survey] = self.cluster.copy()
        apply_extinction_to_photometry(self, model)
        make_unresolved_stars(self, model)
        apply_errors(self, model)
        apply_selection_function(self, model)

        return self.observations[survey]


def calculate_velocity_dispersion_1d(r_50, mass, virial_ratio, eta=10.0):
    """Calculates the 1D velocity dispersion of a cluster given its current virial
    state. See Portegies-Zwart+10 / Emily's thesis for equation derivation

    sigma_1d = [ ( 2 * Q * G * M) / (eta * r_50) ]^0.5

    # Todo: move this
    """
    mass_kg = (mass * u.M_sun).to(u.kg).value
    r_50_m = (r_50 * u.pc).to(u.m).value
    return ((2 * virial_ratio * constants.G.value * mass_kg) / (eta * r_50_m)) ** 0.5
