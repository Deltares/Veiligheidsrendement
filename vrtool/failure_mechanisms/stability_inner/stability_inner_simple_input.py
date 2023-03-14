from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)
import numpy as np


@dataclass
class StabilityInnerSimpleInput:
    safety_factor_2025: np.ndarray
    safety_factor_2075: np.ndarray

    beta_2025: np.ndarray
    beta_2075: np.ndarray
    beta: np.ndarray

    failure_probability_with_elimination: np.ndarray
    failure_probability_elimination: np.ndarray

    is_eliminated: bool
    reliability_calculation_method: ReliabilityCalculationMethod

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> StabilityInnerSimpleInput:
        def _get_valid_bool_value(input_value: Union[str, bool]) -> bool:
            if isinstance(input_value, bool):
                return input_value

            return input_value.lower() == "yes"

        _reliability_calculation_method = None
        _safety_factor_2075 = None
        _beta_2075 = None

        _safety_factor_2025 = mechanism_input.input.get("SF_2025", None)
        _beta_2025 = mechanism_input.input.get("beta_2025", None)
        _beta = mechanism_input.input.get("BETA", None)

        # If all input is defined, the safety factor takes precedence in which
        # reliability calculation method should be used
        if _safety_factor_2025:
            _safety_factor_2075 = mechanism_input.input["SF_2075"]

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
        elif _beta:
            _safety_factor_2025 = None
            _beta_2025 = None

            _reliability_calculation_method = ReliabilityCalculationMethod.BETA_SINGLE
        else:
            raise Exception("Warning: No input values SF or Beta StabilityInner")

        _is_eliminated = mechanism_input.input.get("Elimination", False)
        _failure_probability_elimination = None
        _failure_probability_with_elimination = None
        if _get_valid_bool_value(_is_eliminated):
            _is_eliminated = True
            _failure_probability_elimination = mechanism_input.input["Pf_elim"]
            _failure_probability_with_elimination = mechanism_input.input[
                "Pf_with_elim"
            ]
        elif _is_eliminated:
            raise ValueError("Warning: Elimination defined but not turned on")

        _input = cls(
            safety_factor_2025=_safety_factor_2025,
            safety_factor_2075=_safety_factor_2075,
            beta_2025=_beta_2025,
            beta_2075=_beta_2075,
            beta=_beta,
            reliability_calculation_method=_reliability_calculation_method,
            failure_probability_with_elimination=_failure_probability_with_elimination,
            failure_probability_elimination=_failure_probability_elimination,
            is_eliminated=_is_eliminated,
        )
        return _input
