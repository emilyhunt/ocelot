import numpy as np
from ._base import BaseDifferentialReddeningModel
from ocelot.util.random import fractal_noise_2d
from scipy.interpolate import RegularGridInterpolator


class FractalDifferentialReddening(BaseDifferentialReddeningModel):
    accepts_random_seed = True

    def __init__(self, resolution: int = 256, seed=None) -> None:
        super().__init__()
        self.resolution = resolution
        self._differential_extinction_map = fractal_noise_2d(resolution, seed)
        x = y = np.linspace(0, 1, num=self.resolution)
        self._differential_extinction_interpolator = RegularGridInterpolator(
            (x, y), self._differential_extinction_map, method="linear"
        )

    def _differential_extinction(
        self, x: np.ndarray, y: np.ndarray, width: float
    ) -> np.ndarray:
        points = np.vstack((x, y)).T
        return width * self._differential_extinction_interpolator(points)
