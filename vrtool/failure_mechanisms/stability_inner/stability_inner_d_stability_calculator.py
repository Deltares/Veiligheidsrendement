from typing import Tuple

import numpy as np

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class StabilityInnerDStabilityCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, safety_factor: np.array) -> None:
        self._safety_factor = safety_factor

    def calculate(self, year: int) -> Tuple[float, float]:
        """Calculate the reliability and the probability of failure from an input safety factor of a stix file.
        The reliability is insensitive to the year.
        """
        beta = np.min([calculate_reliability(self._safety_factor), 8.0])

        return beta, beta_to_pf(beta)
