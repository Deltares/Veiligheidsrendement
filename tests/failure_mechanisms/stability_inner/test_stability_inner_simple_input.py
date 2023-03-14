import numpy as np
import pytest

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner import StabilityInnerSimpleInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)


class TestStabilityInnerInput:
    def test_from_mechanism_input_with_safety_factor_range_returns_input_with_safety_factor_range(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["SF_2025"] = np.array([0.1], dtype=float)
        mechanism_input.input["SF_2075"] = np.array([0.2], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.SAFETYFACTOR_RANGE
        )
        assert (
            failure_mechanism_input.safety_factor_2025
            == mechanism_input.input["SF_2025"]
        )
        assert (
            failure_mechanism_input.safety_factor_2075
            == mechanism_input.input["SF_2075"]
        )

        assert failure_mechanism_input.beta_2025 is None
        assert failure_mechanism_input.beta_2075 is None

        assert failure_mechanism_input.beta is None

    def test_from_mechanism_input_with_beta_range_returns_input_with_beta_range(self):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["beta_2025"] = np.array([0.1], dtype=float)
        mechanism_input.input["beta_2075"] = np.array([0.2], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.BETA_RANGE
        )
        assert failure_mechanism_input.beta_2025 == mechanism_input.input["beta_2025"]
        assert failure_mechanism_input.beta_2075 == mechanism_input.input["beta_2075"]

        assert failure_mechanism_input.safety_factor_2025 is None
        assert failure_mechanism_input.safety_factor_2075 is None

        assert failure_mechanism_input.beta is None

    def test_from_mechanism_input_with_beta_returns_input_with_beta(self):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["BETA"] = np.array([0.1], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.BETA_SINGLE
        )
        assert failure_mechanism_input.beta == mechanism_input.input["BETA"]

        assert failure_mechanism_input.beta_2025 is None
        assert failure_mechanism_input.beta_2075 is None

        assert failure_mechanism_input.safety_factor_2025 is None
        assert failure_mechanism_input.safety_factor_2075 is None

    def test_from_mechanism_input_without_any_reliability_or_safety_factor_raises_exception(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")

        # Call
        with pytest.raises(Exception) as exception_error:
            StabilityInnerSimpleInput.from_mechanism_input(mechanism_input)

        # Assert
        assert (
            str(exception_error.value)
            == "Warning: No input values SF or Beta StabilityInner"
        )

    def test_from_mechanism_input_with_all_reliability_and_safety_factor_returns_input_with_safety_factor_range(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["SF_2025"] = np.array([0.1], dtype=float)
        mechanism_input.input["SF_2075"] = np.array([0.2], dtype=float)
        mechanism_input.input["beta_2075"] = np.array([0.3], dtype=float)
        mechanism_input.input["beta_2075"] = np.array([0.4], dtype=float)
        mechanism_input.input["BETA"] = np.array([0.5], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.SAFETYFACTOR_RANGE
        )
        assert (
            failure_mechanism_input.safety_factor_2025
            == mechanism_input.input["SF_2025"]
        )
        assert (
            failure_mechanism_input.safety_factor_2075
            == mechanism_input.input["SF_2075"]
        )

        assert failure_mechanism_input.beta_2025 is None
        assert failure_mechanism_input.beta_2075 is None

        assert failure_mechanism_input.beta is None

    def test_from_mechanism_input_with_elimination_returns_input_with_elimination(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["BETA"] = np.array([0.5], dtype=float)
        mechanism_input.input["Elimination"] = "yes"
        mechanism_input.input["Pf_elim"] = np.array([0.2], dtype=float)
        mechanism_input.input["Pf_with_elim"] = np.array([0.3], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert failure_mechanism_input.is_eliminated
        assert (
            failure_mechanism_input.failure_probability_elimination
            == mechanism_input.input["Pf_elim"]
        )
        assert (
            failure_mechanism_input.failure_probability_with_elimination
            == mechanism_input.input["Pf_with_elim"]
        )

    @pytest.mark.parametrize(
        "elimination",
        [
            pytest.param("NotYes", id="Not a valid string"),
            pytest.param("   ", id="Empty string"),
        ],
    )
    def test_from_mechanism_input_with_elimination_without_valid_value_raises_value_error(
        self, elimination: str
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["BETA"] = np.array([0.5], dtype=float)
        mechanism_input.input["Elimination"] = elimination

        # Call
        with pytest.raises(ValueError) as exception_error:
            StabilityInnerSimpleInput.from_mechanism_input(mechanism_input)

        # Assert
        assert (
            str(exception_error.value)
            == "Warning: Elimination defined but not turned on"
        )
