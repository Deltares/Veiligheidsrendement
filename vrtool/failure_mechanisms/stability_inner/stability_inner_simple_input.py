from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)


@dataclass
class StabilityInnerSimpleInput:
    # TODO: VRTOOL-340. This does not support multiple scenarios.
    # We can remove everything from 2025 / 2075 (discussed with PO)

    safety_factor_2025: np.ndarray
    safety_factor_2075: np.ndarray

    beta_2025: np.ndarray
    beta_2075: np.ndarray

    beta: np.ndarray
    scenario_probability: np.ndarray
    probability_of_failure: np.ndarray

    failure_probability_with_elimination: np.ndarray
    failure_probability_elimination: np.ndarray

    is_eliminated: bool
    reliability_calculation_method: ReliabilityCalculationMethod

    def get_failure_probability_from_scenarios(self) -> float:
        return np.sum(
            np.multiply(self.probability_of_failure, self.scenario_probability)
        )

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> StabilityInnerSimpleInput:
        def _get_valid_bool_value(input_value: str | bool) -> bool:
            if isinstance(input_value, bool):
                return input_value

            return input_value.lower() == "yes"

        _reliability_calculation_method = None
        _safety_factor_2075 = None
        _beta_2075 = None

        _safety_factor_2025 = mechanism_input.input.get("sf_2025", None)
        _beta_2025 = mechanism_input.input.get("beta_2025", None)
        _beta = mechanism_input.input.get("beta", None)

        # If all input is defined, the safety factor takes precedence in which
        # reliability calculation method should be used
        if _safety_factor_2025:
            _safety_factor_2075 = mechanism_input.input["sf_2075"]

            _beta_2025 = None
            _beta = None

            _reliability_calculation_method = (
                ReliabilityCalculationMethod.SAFETYFACTOR_RANGE
            )
        elif _beta_2025:
            _beta_2075 = mechanism_input.input["beta_2075"]

            _safety_factor_2025 = None
            _beta = None

            _reliability_calculation_method = ReliabilityCalculationMethod.BETA_RANGE
        elif isinstance(_beta, np.ndarray) or _beta:
            _safety_factor_2025 = None
            _beta_2025 = None
            _reliability_calculation_method = ReliabilityCalculationMethod.BETA_SINGLE
        else:
            raise Exception("Warning: No input values SF or Beta StabilityInner")

        _is_eliminated = mechanism_input.input.get("elimination", False)
        _failure_probability_elimination = None
        _failure_probability_with_elimination = None
        if _get_valid_bool_value(_is_eliminated):
            _is_eliminated = True
            _failure_probability_elimination = mechanism_input.input["pf_elim"]
            _failure_probability_with_elimination = mechanism_input.input[
                "pf_with_elim"
            ]
        elif _is_eliminated:
            raise ValueError("Warning: Elimination defined but not turned on")

        _input = cls(
            safety_factor_2025=_safety_factor_2025,
            safety_factor_2075=_safety_factor_2075,
            beta_2025=_beta_2025,
            beta_2075=_beta_2075,
            beta=_beta,
            scenario_probability=mechanism_input.input.get(
                "P_scenario", np.ndarray([])
            ),
            probability_of_failure=mechanism_input.input.get("Pf", np.ndarray([])),
            reliability_calculation_method=_reliability_calculation_method,
            failure_probability_with_elimination=_failure_probability_with_elimination,
            failure_probability_elimination=_failure_probability_elimination,
            is_eliminated=_is_eliminated,
        )
        return _input
