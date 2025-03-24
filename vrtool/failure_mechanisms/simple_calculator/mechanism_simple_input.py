from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


@dataclass
class MechanismSimpleInput:
    beta: np.ndarray
    scenario_probability: np.ndarray
    initial_probability_of_failure: np.ndarray
    mechanism_reduction_factor: float

    is_eliminated: bool
    reliability_calculation_method: ReliabilityCalculationMethod

    def get_failure_probability_from_scenarios(self) -> float:
        """
        Gets the current failure probability based on the `scenario_probability` and `beta`.
        We use `beta` instead of `initial_probability_of_failure` as the latter remains constant whilst
        `beta` changes throughout the different steps of an optimization since imported from the database.

        Returns:
            float: Failure probability as the inner product of `beta` and `scenario_probability`.
        """
        _probability_single_assessment = beta_to_pf(self.beta)
        return np.sum(
            np.multiply(_probability_single_assessment, self.scenario_probability)
        )

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> MechanismSimpleInput:
        """
        Generates a `StabilityInnerSimpleInput` object based on the provided `MechanismInput`.

        Args:
            mechanism_input (MechanismInput): Mechanism input containing all the required input data.

        Returns:
            StabilityInnerSimpleInput: Resulting mapped object.
        """

        def _get_valid_bool_value(input_value: str | bool) -> bool:
            if isinstance(input_value, bool):
                return input_value

            return input_value.lower() == "yes"

        _reliability_calculation_method = None
        _beta = mechanism_input.input.get("beta", None)

        # If all input is defined, the safety factor takes precedence in which
        # reliability calculation method should be used
        if isinstance(_beta, np.ndarray) or _beta:
            _reliability_calculation_method = ReliabilityCalculationMethod.BETA_SINGLE
        else:
            raise ValueError("Warning: No input values SF or Beta StabilityInner")

        _is_eliminated = _get_valid_bool_value(
            mechanism_input.input.get("elimination", False)
        )

        _prob_solution_failure = mechanism_input.input.get("Pf", np.ndarray([]))
        _input = cls(
            beta=_beta,
            scenario_probability=mechanism_input.input.get(
                "P_scenario", np.ndarray([])
            ),
            initial_probability_of_failure=_prob_solution_failure,
            mechanism_reduction_factor=mechanism_input.input.get(
                "piping_reduction_factor", 1
            ),
            reliability_calculation_method=_reliability_calculation_method,
            is_eliminated=_is_eliminated,
        )
        return _input
