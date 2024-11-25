from ._base import BaseDifferentialReddeningModel


class FractalDifferentialReddening(BaseDifferentialReddeningModel):
    def __init__(self, resolution: int = 256) -> None:
        super().__init__()
        self.resolution = resolution