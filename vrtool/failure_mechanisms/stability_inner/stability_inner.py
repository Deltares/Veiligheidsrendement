import numpy as np
from typing import Tuple
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta
from vrtool.flood_defence_system.mechanism_input import MechanismInput


class StabilityInner:
    """
    Contains all methods related to performing a stability inner calculation.
    """

    __model_factor: float = 1.06

    def calculate_reliability(safety_factor: np.ndarray) -> float:
        """
        Calculates the reliability (beta) based on the input arguments.
        Args:
            safety_factor (float): the safety factor to calculate the reliability with.
        Returns:
            float: the safety factor.
        """
        beta = ((safety_factor.item() / StabilityInner.__model_factor) - 0.41) / 0.15
        beta = np.min([beta, 8.0])
        return beta

    def calculate_safety_factor(reliability: float) -> float:
        """
        Calculates the safety factor based on the input arguments.
        Args:
            reliability (float): the reliability (beta) to calculate the safety factor with.
        Returns:
            float: the safety factor.
        """
        return (0.41 + 0.15 * reliability) * StabilityInner.__model_factor

    def calculate_simple(
        mechanism_input: MechanismInput, year: int
    ) -> Tuple[float, float]:

        if "SF_2025" in mechanism_input.input:
            # Simple interpolation of two safety factors and translation to a value of beta at 'year'.
            # In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
            safety_factor_interpolate_function = interpolate.interp1d(
                [0, 50],
                np.array(
                    [mechanism_input.input["SF_2025"], mechanism_input.input["SF_2075"]]
                ).flatten(),
                fill_value="extrapolate",
            )
            safety_factor = safety_factor_interpolate_function(year)
            beta = np.min([StabilityInner.calculate_reliability(safety_factor), 8.0])
        elif "beta_2025" in mechanism_input.input:

            beta_interpolate_function = interpolate.interp1d(
                [0, 50],
                np.array(
                    [
                        mechanism_input.input["beta_2025"],
                        mechanism_input.input["beta_2075"],
                    ]
                ).flatten(),
                fill_value="extrapolate",
            )

            beta = beta_interpolate_function(year)
            beta = np.min([beta, 8])
        elif (
            "BETA" in mechanism_input.input
        ):  # situation where beta is constant in time
            beta = np.min([mechanism_input.input["BETA"].item(), 8.0])
        else:
            raise Exception("Warning: No input values SF or Beta StabilityInner")
        # Check if there is an elimination measure present (diaphragm wall)
        if "Elimination" in mechanism_input.input.keys():
            if mechanism_input.input["Elimination"] == "yes":
                # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                # addition: should not be more unsafe
                failure_probability = np.min(
                    [
                        beta_to_pf(beta) * mechanism_input.input["Pf_elim"]
                        + mechanism_input.input["Pf_with_elim"]
                        * (1 - mechanism_input.input["Pf_elim"]),
                        beta_to_pf(beta),
                    ]
                )
                beta = pf_to_beta(failure_probability)
            else:
                raise ValueError("Warning: Elimination defined but not turned on")

        return [beta, beta_to_pf(beta)]
