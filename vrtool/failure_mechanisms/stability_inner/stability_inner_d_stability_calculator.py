from typing import Tuple

import numpy as np

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.stability_inner.stability_inner_d_stability_input import (
    StabilityInnerDStabilityInput,
)
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class StabilityInnerDStabilityCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, mechanism_input: StabilityInnerDStabilityInput) -> None:
        self._mechanism_input = mechanism_input

    def calculate(self, year: int) -> Tuple[float, float]:
        """Calculate the reliability and the probability of failure from an input safety factor of a stix file.
        The reliability is insensitive to the year.
        """
        beta = np.min([calculate_reliability(self._mechanism_input.safety_factor), 8.0])

        return beta, beta_to_pf(beta)
