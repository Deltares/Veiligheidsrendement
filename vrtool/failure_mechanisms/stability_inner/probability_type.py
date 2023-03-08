from enum import Enum


class ProbabilityType(Enum):
    """
    Definition of the different types of probability definitions.
    """

    SAFETYFACTOR_RANGE = 0
    BETA_RANGE = 1
    BETA_SINGLE = 2
