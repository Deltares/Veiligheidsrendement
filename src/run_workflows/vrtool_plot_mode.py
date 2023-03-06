from enum import Enum


class VrToolPlotMode(Enum):
    """
    Definition of the different types of plots supported during a run of a `VrtoolRunProtocol` instance.
    """

    STANDARD = 0
    EXTENSIVE = 1
