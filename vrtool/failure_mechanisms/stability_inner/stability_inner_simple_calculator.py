import numpy as np
from scipy import interpolate

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta
from vrtool.failure_mechanisms.stability_inner.stability_inner_simple_input import (
    StabilityInnerSimpleInput,
)
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)

from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
)


class StabilityInnerSimpleCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, mechanism_input: StabilityInnerSimpleInput) -> None:
        if not isinstance(mechanism_input, StabilityInnerSimpleInput):
            raise ValueError(
                "Expected instance of a {}.".format(StabilityInnerSimpleInput.__name__)
            )

        if not self._is_supported_reliability_calculation_method(
            mechanism_input.reliability_calculation_method
        ):
            raise ValueError(
                "Unsupported value of {}.".format(ReliabilityCalculationMethod.__name__)
            )

        self._mechanism_input = mechanism_input

    def _is_supported_reliability_calculation_method(
        self, calculation_method: ReliabilityCalculationMethod
    ):
        match calculation_method:
            case ReliabilityCalculationMethod.SAFETYFACTOR_RANGE:
                return True
            case ReliabilityCalculationMethod.BETA_RANGE:
                return True
            case ReliabilityCalculationMethod.BETA_SINGLE:
                return True
            case _:
                return False

    def calculate(self, year: int) -> tuple[float, float]:
        match self._mechanism_input.reliability_calculation_method:
            case ReliabilityCalculationMethod.SAFETYFACTOR_RANGE:
                # Simple interpolation of two safety factors and translation to a value of beta at 'year'.
                # In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
                safety_factor_interpolate_function = interpolate.interp1d(
                    [0, 50],
                    np.array(
                        [
                            self._mechanism_input.safety_factor_2025,
                            self._mechanism_input.safety_factor_2075,
                        ]
                    ).flatten(),
                    fill_value="extrapolate",
                )
                safety_factor = safety_factor_interpolate_function(year)
                beta = np.min([calculate_reliability(safety_factor), 8.0])

            case ReliabilityCalculationMethod.BETA_RANGE:
                beta_interpolate_function = interpolate.interp1d(
                    [0, 50],
                    np.array(
                        [
                            self._mechanism_input.beta_2025,
                            self._mechanism_input.beta_2075,
                        ]
                    ).flatten(),
                    fill_value="extrapolate",
                )

                beta = beta_interpolate_function(year)
                beta = np.min([beta, 8])

            case ReliabilityCalculationMethod.BETA_SINGLE:
                # situation where beta is constant in time
                beta = np.min([self._mechanism_input.beta.item(), 8.0])

        # Check if there is an elimination measure present (diaphragm wall)
        if self._mechanism_input.is_eliminated:
            # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
            # addition: should not be more unsafe
            failure_probability = np.min(
                [
                    beta_to_pf(beta)
                    * self._mechanism_input.failure_probability_elimination
                    + self._mechanism_input.failure_probability_with_elimination
                    * (1 - self._mechanism_input.failure_probability_elimination),
                    beta_to_pf(beta), 
                ]
            )
            beta = pf_to_beta(failure_probability)

        return [beta, beta_to_pf(beta)]
