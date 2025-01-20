"""Definition of a base SimualtedCluster class that handles checking and grouping of
parameters and data.
"""

from __future__ import annotations
import numpy as np
from astropy import units as u
from astropy import constants
from astropy.coordinates import SkyCoord
import pandas as pd
from ocelot.simulate.errors import NotEnoughStarsError
from ocelot.simulate.photometry import create_population, apply_extinction
from ocelot.simulate.astrometry import generate_true_star_astrometry
from ocelot.simulate.binaries import make_binaries
from ocelot.simulate.observation import (
    apply_extinction_to_photometry,
    make_unresolved_stars,
    apply_photometric_errors,
    apply_astrometric_errors,
    apply_selection_function,
    cleanup_observation,
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
    """Class for keeping track of parameters specified for a cluster to simulate.

    Parameters
    ----------
    position : SkyCoord
        Position of the cluster as an astropy SkyCoord. Must have full 3D distance and
        3D velocity information.
    mass : float
        Mass of the cluster in solar masses.
    log_age : float
        Age of the cluster in log (base 10) years.
    metallicity : float
        The metallicity of the cluster, [Fe/H].
    r_core : float
        The core radius of the cluster, in parsecs.
    r_tidal : float
        The tidal radius of the cluster, in parsecs.
    extinction : float
        The extinction (A_V / A_0) of the cluster in magnitudes.
    differential_extinction : float, optional
        Amount of differential extinction to apply to the cluster, also in magnitudes.
        Default: 0.
    minimum_stars : int, optional
        Specify the minimum number of stars the cluster can have. Default: 0
    virial_ratio : float, optional
        Virial ratio of the cluster. Acts as a square-root scale factor to the cluster's
        velocity dispersion. Default: 0.5, meaning that the cluster is virialized.
    eta_virial_ratio : float, optional
        Scale factor of the 1D velocity dispersion equation. Default: 10, which is a
        good approximation for most clusters.
    id : int, optional
        ID of the simulated cluster. When set, this allows for unique identification of
        different simulated clusters. Default: 0.

    Attributes
    ----------
    r_50 : float
        The half-light radius of the cluster in parsecs.
    velocity_dispersion_1d : float
        The 1D velocity dispersion of the cluster in metres per second.
    """

    # Todo: all of these should really be astropy quantity objects. Units!!!
    position: SkyCoord
    mass: float
    log_age: float
    metallicity: float
    r_core: float  # Todo these should be optional, so that e.g. a Plummer profile can be supported
    r_tidal: float  # Todo these should be optional, so that e.g. a Plummer profile can be supported
    extinction: float
    differential_extinction: float = 0.0
    minimum_stars: int = 1  # Todo does this do anything?
    virial_ratio: float = 0.5
    eta_virial_ratio: float = 10.0
    id: int = 0

    # The following fields are calculated in __post_init__, as they depend on initial
    # values:
    r_50: float = field(init=False)
    velocity_dispersion_1d: float = field(init=False)
    n_stars: int = field(init=False, default=0)  # Todo remove?

    # Todo the below should probably be phased out in favour of @property values
    ra: float = field(init=False)
    dec: float = field(init=False)
    l: float = field(init=False)  # noqa: E741
    b: float = field(init=False)
    distance: float = field(init=False)
    pmra: float = field(init=False)
    pmdec: float = field(init=False)
    radial_velocity: float = field(init=False)

    def __post_init__(self):
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
    """Class for keeping track of all models that a generated SimulatedCluster will use.

    Parameters
    ----------
    distribution : BaseClusterDistributionModel or None, optional
        The distribution model for the cluster. Must be an instance of
        BaseClusterDistributionModel. Default: None, meaning that a King62 model is used
    binaries : BaseBinaryStarModel or None, optional
        The binary star model for the cluster. Must be an instance of
        BaseBinaryStarModel. Default: None, meaning that a
        MoeDiStefanoMultiplicityRelation (with Duchene-Kraus+13 below 1 MSun) is used.
    differential_reddening : BaseDifferentialReddeningModel or None, optional
        The differential reddening model for the cluster. Must be an instance of
        BaseDifferentialReddeningModel. Default: None, meaning that a
        FractalDifferentialReddening model is used.
    observations : list or tuple of BaseObservation, optional
        A list or tuple of observation models for the cluster. Each model must be an
        instance of a BaseObservation. These observation models will be iterated through
        when generating cluster observations to generate as many (or few) observation
        simulations of a cluster as you'd like. Observation models must be unique.
        Default: None, meaning that no cluster observation simulation will be generated.

    Attributes
    ----------
    observations_dict : dict of BaseObservation models
        Same as the observations parameter, but with observations instead organized into
        a dictionary.
    """

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

        This method is called during simulated cluster generation and should not need
        to be used by users.

        Parameters
        ----------
        parameters : SimulatedClusterParameters
            The parameters of the cluster to simulate.
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
        """Return an instance of a SimulatedClusterModels model with default options.

        Parameters
        ----------
        parameters : SimulatedClusterParameters
            The parameters of the cluster to simulate.
        seed : int
            Random seed to use for default models that incorporate randomness.

        Returns
        -------
        SimulatedClusterModels
            An instance of SimulatedClusterModels with default options already set up.
        """
        return SimulatedClusterModels().initialise_defaults(
            parameters=parameters, seed=seed
        )


@dataclass
class SimulatedClusterFeatures:
    """Class for keeping track of all features used to simulate a cluster.

    This class mostly exists to aid in testing parts of ocelot.simulate with certain
    physical effects turned on or off.

    Parameters
    ----------
    binary_stars : bool, optional
        Whether or not to simulate binary stars in the cluster. Default: True
    differential_extinction : bool, optional
        Whether or not to simulate differential extinction of the cluster. Default: True
    selection_effects : bool, optional
        Whether or not to simulate selection effects in simulated observations of the
        cluster. Default: True
    astrometric_uncertainties : bool, optional
        Whether or not to apply astrometric uncertainties to observations of the
        cluster. Default: True
    photometric_uncertainties : bool, optional
        Whether or not to apply photometric uncertainties to observations of the
        cluster. Default: True
    """

    # Intrinsic (i.e. impact how generation is done)
    binary_stars: bool = True

    # Extrinsic (i.e. impact observations of the simulated cluster)
    differential_extinction: bool = True
    selection_effects: bool = True
    astrometric_uncertainties: bool = True
    photometric_uncertainties: bool = True


class SimulatedCluster:
    """A class for simulating and keeping track of a simulated cluster - including its
    original membership list and any observations simulated from it.
    
    This class is the main entry point in ocelot for simulating star clusters.

    Parameters
    ----------
    parameters : SimulatedClusterParameters or dict
        Parameters of the simulated cluster to generate. Should be a
        SimulatedClusterParameters object, but may also be a dict with keys for required
        parameters such as position, etc.
    models : SimulatedClusterModels, dict or None, optional
        SimulatedClusterModels object or dict containing models used to overwrite or
        augment certain simulation features. Default: None
    prune_simulated_cluster : str, optional
        Optional string used early during cluster simulation to prune a simulated
        cluster. Will be passed to pandas.DataFrame.query(). It can access parameters
        read directly from cluster isochrones, including magnitude, temperature,
        and luminosity. Default: ""
    random_seed : int or None, optional
        Random seed to use for cluster generation. When set, cluster generation with the
        same seed should be identical. Default: None
    features : SimulatedClusterFeatures or dict or None, optional
        A SimulatedClusterFeatures or dict object specifying features of cluster
        generation to turn off. Mostly intended to aid with testing. Default: None
    observations : list of str or None, optional
        List of observations to generate. Soon to be deprecated; do not use.

    Attributes
    ----------
    isochrone : pd.DataFrame
        Dataframe containing the isochrone used to simulate this cluster.
    cluster : pd.DataFrame
        Dataframe containing the true members of the cluster.
    observations : dict of pd.DataFrame
        Dict of dataframes, with each one containing a different observation of the
        same cluster.
    """

    def __init__(
        self,
        parameters: SimulatedClusterParameters | dict,
        models: SimulatedClusterModels | dict | None = None,
        prune_simulated_cluster: str = "",
        random_seed: int | None = None,
        features: SimulatedClusterFeatures | dict | None = None,
        observations: list[str] | None = None,  # Todo consider removing
    ):
        # Stuff for handling randomness
        if random_seed is None:
            # Select a random seed from 0 to the largest possible signed 64 bit int
            # This enforces seeding even when a user doesn't specify a seed - helping
            # to ensure reproducibility.
            random_seed = np.random.default_rng().integers(2**32 - 1)

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

        # Handle features switches
        if features is None:
            features = SimulatedClusterFeatures()
        if isinstance(features, dict):
            features = SimulatedClusterFeatures(**features)
        self.features: SimulatedClusterFeatures = features

        # Handle pruning
        self.prune_simulated_cluster = prune_simulated_cluster

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
        self.stars: int = 0  # Todo remove
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
        """Makes entire cluster according to specification set at initialization.

        This is the main function that should be used to simulate a cluster.
        
        Returns
        -------
        SimulatedCluster
            A reference to the SimulatedCluster object.
        """
        self.make_cluster()
        self.make_observations()
        return self

    def make_cluster(self):
        """Creates the true stars and positions in a cluster.

        In general, just calling .make() is the recommended method for most users.
        
        Returns
        -------
        SimulatedCluster
            A reference to the SimulatedCluster object.
        """
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
        return self

    def make_observations(self):
        """Makes all observations of the cluster.

        In general, just calling .make() is the recommended method for most users.

        Returns
        -------
        SimulatedCluster
            A reference to the SimulatedCluster object."""
        for observation in self._observations_to_make:
            self.make_observation(observation)
        self._observations_generated = True
        return self

    def make_observation(self, survey: str, seed=None):
        """Makes one observation of the cluster.

        In general, just calling .make() is the recommended method for most users.

        Parameters
        ----------
        survey : str
            Name of the survey (i.e. name in self.observations) to make.
        seed : None, optional
            Seed used to reseed the random generator. Useful for doing multiple
            different simulated observations of the same cluster. May not be supported
            by all cluster observation models. Default: None, meaning that the current
            cluster random number generator generated from the seed specified during
            class initialization is used.
        
        Returns
        -------
        pd.DataFrame
            The simulated cluster observation made by this method. 
        """
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
        apply_photometric_errors(self, model)
        apply_astrometric_errors(self, model)
        apply_selection_function(self, model)
        cleanup_observation(self, model)

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
