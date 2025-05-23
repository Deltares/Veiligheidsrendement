import numpy as np
import pytest

from vrtool.failure_mechanisms import MechanismInput
from vrtool.failure_mechanisms.simple_calculator import MechanismSimpleInput
from vrtool.failure_mechanisms.stability_inner.reliability_calculation_method import (
    ReliabilityCalculationMethod,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class TestMechanismSimpleInput:
    def test_from_mechanism_input_with_beta_returns_input_with_beta(
        self, mechanism_input_fixture: MechanismInput
    ):
        # Setup
        assert isinstance(mechanism_input_fixture, MechanismInput)

        # Call
        failure_mechanism_input = MechanismSimpleInput.from_mechanism_input(
            mechanism_input_fixture
        )

        # Assert
        assert (
            failure_mechanism_input.reliability_calculation_method
            == ReliabilityCalculationMethod.BETA_SINGLE
        )
        assert failure_mechanism_input.beta == mechanism_input_fixture.input["beta"]
        assert failure_mechanism_input.mechanism_reduction_factor == 1

    def test_from_mechanism_input_without_any_reliability_raises_exception(
        self,
    ):
        # Setup
        mechanism_input = MechanismInput("")

        # Call
        with pytest.raises(Exception) as exception_error:
            MechanismSimpleInput.from_mechanism_input(mechanism_input)

        # Assert
        assert (
            str(exception_error.value)
            == "Warning: No input values SF or Beta StabilityInner"
        )

    def test_from_mechanism_input_with_elimination_returns_input_with_elimination(
        self, mechanism_input_fixture: MechanismInput
    ):
        # Setup
        mechanism_input_fixture.input["elimination"] = "yes"
        mechanism_input_fixture.input["pf_elim"] = np.array([0.2], dtype=float)
        mechanism_input_fixture.input["pf_with_elim"] = np.array([0.3], dtype=float)

        # Call
        failure_mechanism_input = MechanismSimpleInput.from_mechanism_input(
            mechanism_input_fixture
        )

        # Assert
        assert failure_mechanism_input.is_eliminated

    @pytest.mark.parametrize(
        "initial_probability_of_failure, scenario_probability, expected_result",
        [
            pytest.param([0.1, 1], [0.2, 0.02], 0.04),
            pytest.param([0.42, 0.24], [0.24, 0.42], 0.2015),
            pytest.param([1, 0.1], [0.02, 0.2], 0.04),
        ],
    )
    def test_given_initial_pf_and_scenario_probability_as_arrays_then_returns_expectation(
        self,
        initial_probability_of_failure: list[float],
        scenario_probability: list[float],
        expected_result: float,
    ):
        # 1. Define test data.
        _dummy_input = MechanismSimpleInput(
            beta=pf_to_beta(np.array(initial_probability_of_failure)),
            scenario_probability=np.array(scenario_probability),
            initial_probability_of_failure=np.array(initial_probability_of_failure),
            is_eliminated=False,
            reliability_calculation_method=ReliabilityCalculationMethod.BETA_SINGLE,
            mechanism_reduction_factor=np.array([1000] * len(scenario_probability)),
        )

        # 2. Run test.
        _result = _dummy_input.get_failure_probability_from_scenarios()

        # 3. Verify expectations.
        assert _result == pytest.approx(expected_result, 1e-3)
