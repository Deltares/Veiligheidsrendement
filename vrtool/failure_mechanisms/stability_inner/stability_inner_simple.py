import numpy as np
from scipy import interpolate

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


class StabilityInnerSimple:
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def calculate(
        mechanism_input: StabilityInnerSimpleInput, year: int
    ) -> tuple[float, float]:
        """Calculates the reliability and the the probability of failure based on its input arguments.

        Args:
            mechanism_input (StabilityInnerSimpleInput): The input to calculate the reliability and the probability of failure with.
            year (int): The year to calculate the reliability and the probability of failure for.

        Returns:
            tuple[float, float]: A tuple containing the reliability and the probability of failure.
        """

        match mechanism_input.reliability_calculation_method:
            case ReliabilityCalculationMethod.SAFETYFACTOR_RANGE:
                # Simple interpolation of two safety factors and translation to a value of beta at 'year'.
                # In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
                safety_factor_interpolate_function = interpolate.interp1d(
                    [0, 50],
                    np.array(
                        [
                            mechanism_input.safety_factor_2025,
                            mechanism_input.safety_factor_2075,
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
                            mechanism_input.beta_2025,
                            mechanism_input.beta_2075,
                        ]
                    ).flatten(),
                    fill_value="extrapolate",
                )

                beta = beta_interpolate_function(year)
                beta = np.min([beta, 8])

            case ReliabilityCalculationMethod.BETA_SINGLE:
                # situation where beta is constant in time
                beta = np.min([mechanism_input.beta.item(), 8.0])

        # Check if there is an elimination measure present (diaphragm wall)
        if mechanism_input.is_eliminated:
            # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
            # addition: should not be more unsafe
            failure_probability = np.min(
                [
                    beta_to_pf(beta) * mechanism_input.failure_probability_elimination
                    + mechanism_input.failure_probability_with_elimination
                    * (1 - mechanism_input.failure_probability_elimination),
                    beta_to_pf(beta),
                ]
            )
            beta = pf_to_beta(failure_probability)

        return [beta, beta_to_pf(beta)]
