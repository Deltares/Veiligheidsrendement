import numpy as np

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.simple_calculator.mechanism_simple_input import (
    MechanismSimpleInput,
)
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    BETA_THRESHOLD,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class MechanismSimpleCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, mechanism_input: MechanismSimpleInput) -> None:
        if not isinstance(mechanism_input, MechanismSimpleInput):
            raise ValueError(
                "Expected instance of a {}.".format(MechanismSimpleInput.__name__)
            )

        ReliabilityCalculationMethod.is_valid(
            mechanism_input.reliability_calculation_method
        )

        self._mechanism_input = mechanism_input

    def calculate(self, year: int) -> tuple[float, float]:
        # situation where beta is constant in time
        _pf = self._mechanism_input.get_failure_probability_from_scenarios()
        beta = np.min([pf_to_beta(_pf), BETA_THRESHOLD])

        # Check if there is an elimination measure present (diaphragm wall)
        if self._mechanism_input.is_eliminated:
            # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
            # addition: should not be more unsafe
            failure_probability = np.min(
                [
                    beta_to_pf(beta) / self._mechanism_input.mechanism_reduction_factor,
                    beta_to_pf(beta),
                ]
            )
            beta = pf_to_beta(failure_probability)

        return (beta, beta_to_pf(beta))
