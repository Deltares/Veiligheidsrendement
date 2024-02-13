import numpy as np
import pytest

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner import StabilityInnerSimpleInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)


class TestStabilityInnerSimpleInput:
    def test_from_mechanism_input_with_safety_factor_range_returns_input_with_safety_factor_range(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["sf_2025"] = np.array([0.1], dtype=float)
        mechanism_input.input["sf_2075"] = np.array([0.2], dtype=float)

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
            == mechanism_input.input["sf_2025"]
        )
        assert (
            failure_mechanism_input.safety_factor_2075
            == mechanism_input.input["sf_2075"]
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
        mechanism_input.input["beta"] = np.array([0.1], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.BETA_SINGLE
        )
        assert failure_mechanism_input.beta == mechanism_input.input["beta"]

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
        mechanism_input.input["sf_2025"] = np.array([0.1], dtype=float)
        mechanism_input.input["sf_2075"] = np.array([0.2], dtype=float)
        mechanism_input.input["beta_2075"] = np.array([0.3], dtype=float)
        mechanism_input.input["beta_2075"] = np.array([0.4], dtype=float)
        mechanism_input.input["beta"] = np.array([0.5], dtype=float)

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
            == mechanism_input.input["sf_2025"]
        )
        assert (
            failure_mechanism_input.safety_factor_2075
            == mechanism_input.input["sf_2075"]
        )

        assert failure_mechanism_input.beta_2025 is None
        assert failure_mechanism_input.beta_2075 is None

        assert failure_mechanism_input.beta is None

    def test_from_mechanism_input_with_elimination_returns_input_with_elimination(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["beta"] = np.array([0.5], dtype=float)
        mechanism_input.input["elimination"] = "yes"
        mechanism_input.input["pf_elim"] = np.array([0.2], dtype=float)
        mechanism_input.input["pf_with_elim"] = np.array([0.3], dtype=float)

        # Call
        failure_mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert failure_mechanism_input.is_eliminated
        assert (
            failure_mechanism_input.failure_probability_elimination
            == mechanism_input.input["pf_elim"]
        )
        assert (
            failure_mechanism_input.failure_probability_with_elimination
            == mechanism_input.input["pf_with_elim"]
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
        mechanism_input.input["beta"] = np.array([0.5], dtype=float)
        mechanism_input.input["elimination"] = elimination

        # Call
        with pytest.raises(ValueError) as exception_error:
            StabilityInnerSimpleInput.from_mechanism_input(mechanism_input)

        # Assert
        assert (
            str(exception_error.value)
            == "Warning: Elimination defined but not turned on"
        )

    @pytest.mark.parametrize(
        "probability_of_failure, scenario_probability, expected_result",
        [
            pytest.param([0.1, 1], [0.2, 0.02], 0.04),
            pytest.param([0.42, 0.24], [0.24, 0.42], 0.2015),
            pytest.param([1, 0.1], [0.02, 0.2], 0.04),
        ],
    )
    def test_given_probability_of_failure_and_scenario_probability_as_arrays_then_returns_expectation(
        self,
        probability_of_failure: list[float],
        scenario_probability: list[float],
        expected_result: float,
    ):
        # 1. Define test data.
        _dummy_input = StabilityInnerSimpleInput(
            safety_factor_2025=np.array([]),
            safety_factor_2075=np.array([]),
            beta_2025=np.array([]),
            beta_2075=np.array([]),
            beta=np.array([]),
            scenario_probability=np.array(scenario_probability),
            probability_of_failure=np.array(probability_of_failure),
            failure_probability_with_elimination=np.array([]),
            failure_probability_elimination=np.array([]),
            is_eliminated=False,
            reliability_calculation_method=ReliabilityCalculationMethod.BETA_SINGLE,
        )

        # 2. Run test.
        _result = _dummy_input.get_failure_probability_from_scenarios()

        # 3. Verify expectations.
        assert _result == pytest.approx(expected_result, 1e-3)
