import numpy as np
from scipy import interpolate

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.general.generic_failure_mechanism_input import (
    GenericFailureMechanismInput,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class GenericFailureMechanismCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, mechanism_input: GenericFailureMechanismInput) -> None:
        if not isinstance(mechanism_input, GenericFailureMechanismInput):
            raise ValueError(
                "Expected instance of a {}.".format(
                    GenericFailureMechanismInput.__name__
                )
            )

        self._mechanism_input = mechanism_input

    def calculate(self, year: int) -> tuple[float, float]:
        betat = interpolate.interp1d(
            self._mechanism_input.time_grid,
            self._mechanism_input.beta_grid,
            fill_value="extrapolate",
        )
        beta = float(betat(year))

        return (beta, beta_to_pf(beta))
