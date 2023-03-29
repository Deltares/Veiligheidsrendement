from dataclasses import dataclass

import numpy as np

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol

# TODO TO put in another file
@dataclass
class StabilityInnerDStabilityInput:
    safety_factor_2025: np.ndarray

    # safety_factor_2075: np.ndarray
    #
    # beta_2025: np.ndarray
    # beta_2075: np.ndarray
    # beta: np.ndarray

    @classmethod
    def from_stix_input(
            cls,
            stix_input: str,
    ):
        _input = cls(
            safety_factor_2025=np.array([1, 2, 3]),
        )
        return _input


class StabilityInnerDStabilityCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, mechanism_input: StabilityInnerDStabilityInput) -> None:
        pass
