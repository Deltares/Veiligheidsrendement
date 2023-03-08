from __future__ import annotations

from dataclasses import dataclass
from vrtool.flood_defence_system.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.probability_type import ProbabilityType
import numpy as np


@dataclass
class StabilityInnerInput:
    safety_factor_2025: np.ndarray
    safety_factor_2075: np.ndarray

    beta_2025: np.ndarray
    beta_2075: np.ndarray
    beta: np.ndarray

    failure_probability_with_elimination: np.ndarray
    failure_probability_elimination: np.ndarray

    is_eliminated: bool
    probability_type: ProbabilityType

    @classmethod
    def from_mechanism_input(cls, input: MechanismInput) -> StabilityInnerInput:
        _probability_type = None
        _safety_factor_2075 = None
        _beta_2075 = None

        _safety_factor_2025 = input.input.get("SF_2025", None)
        _beta_2025 = input.input.get("beta_2025", None)
        _beta = input.input.get("BETA", None)

        # If all input is defined, the safety factor takes precedence in which
        # reliability calculation method should be used
        if _safety_factor_2025:
            _safety_factor_2075 = input.input["SF_2075"]
            _probability_type = ProbabilityType.SAFETYFACTOR_RANGE
        elif _beta_2025:
            _beta_2075 = input.input["beta_2075"]
            _probability_type = ProbabilityType.BETA_RANGE
        elif _beta:
            _probability_type = ProbabilityType.BETA_SINGLE
        else:
            raise Exception("Warning: No input values SF or Beta StabilityInner")

        _is_eliminated = input.input.get("Elimination", None)
        _failure_probability_elimination = None
        _failure_probability_with_elimination = None
        if _is_eliminated == "yes":
            _is_eliminated = True
            _failure_probability_elimination = input.input["Pf_elim"]
            _failure_probability_with_elimination = input.input["Pf_with_elim"]
        elif _is_eliminated:
            raise ValueError("Warning: Elimination defined but not turned on")

        _input = cls(
            safety_factor_2025=_safety_factor_2025,
            safety_factor_2075=_safety_factor_2075,
            beta_2025=_beta_2025,
            beta_2075=_beta_2075,
            beta=_beta,
            probability_type=_probability_type,
            failure_probability_with_elimination=_failure_probability_with_elimination,
            failure_probability_elimination=_failure_probability_elimination,
            is_eliminated=_is_eliminated,
        )
        return _input
