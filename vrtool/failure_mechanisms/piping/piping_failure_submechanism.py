from enum import Enum


class PipingFailureSubmechanism(Enum):
    """
    Definition of the different piping failure submechanisms.
    """

    PIPING = 0
    HEAVE = 1
    UPLIFT = 2
