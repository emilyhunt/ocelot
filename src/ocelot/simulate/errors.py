"""Defines a set of errors raised by the package."""


class ClusterSimulationError(Exception):
    pass


class NotEnoughStarsError(ClusterSimulationError):
    pass


class CoreRadiusTooLargeError(ClusterSimulationError):
    pass
