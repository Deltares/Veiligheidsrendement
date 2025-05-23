from typing import Iterator

import pytest

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.piping import PipingSemiProbabilisticCalculator
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class TestPipingSemiProbabilisticCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        _calculator = PipingSemiProbabilisticCalculator(
            _mechanism_input, _load, 0, DikeTrajectInfo(traject_name="")
        )

        # Assert
        assert isinstance(_calculator, PipingSemiProbabilisticCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_mechanism_input(self):
        # Setup
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                "NotMechanismInput", _load, 0, DikeTrajectInfo(traject_name="")
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a MechanismInput."

    def test_init_with_invalid_load_input(self):
        # Setup
        _mechanism_input = MechanismInput("")

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input, "NotLoad", 0, DikeTrajectInfo(traject_name="")
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a LoadInput."

    def test_init_with_invalid_initial_year(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input,
                _load,
                "NotInitialYear",
                DikeTrajectInfo(traject_name=""),
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a int."

    def test_init_with_invalid_trajec_info(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input, _load, 0, "NotATrajectInfo"
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a DikeTrajectInfo."

    @pytest.fixture(name="valid_piping_calculator")
    def _get_valid_piping_calculator_fixture(
        self,
    ) -> Iterator[PipingSemiProbabilisticCalculator]:
        """
        Fixture to retrieve an instance of a `PipingSemiProbabilisticCalculator`
        to be used as default test data.

        Returns:
            PipingSemiProbabilisticCalculator: Valid calculator instance.
        """
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        yield PipingSemiProbabilisticCalculator(
            _mechanism_input, _load, 0, DikeTrajectInfo(traject_name="")
        )

    @pytest.fixture(name="scenario_pf_args")
    def _get_scenario_pf_args_fixture(self) -> Iterator[tuple[float, dict]]:
        _beta = 4.2
        _mechanism_input_dict = {
            "piping_reduction_factor": 1000,
            # The following values will simply allow to compute `beta_to_pf(scenario_beta)`
            "pf_elim": 0,
            "pf_with_elim": 1,
        }
        yield _beta, _mechanism_input_dict

    def test_get_scenario_pf_with_piping_reduction_factor(
        self,
        scenario_pf_args: tuple[float, dict],
        valid_piping_calculator: PipingSemiProbabilisticCalculator,
    ):
        # 1. Define test data.
        _beta, _mechanism_input_dict = scenario_pf_args
        assert "piping_reduction_factor" in _mechanism_input_dict

        # 2. Run test.
        _computed_pf = valid_piping_calculator._get_scenario_pf(
            _beta, _mechanism_input_dict
        )

        # 3. Verify expectations.
        assert _computed_pf == pytest.approx(1.3345e-08, rel=1e-4)

    def test_get_scenario_pf_without_piping_reduction_factor(
        self,
        scenario_pf_args: tuple[float, dict],
        valid_piping_calculator: PipingSemiProbabilisticCalculator,
    ):
        # 1. Define test data.
        _beta, _mechanism_input_dict = scenario_pf_args
        if "piping_reduction_factor" in _mechanism_input_dict:
            _mechanism_input_dict.pop("piping_reduction_factor")

        # 2. Run test.
        _computed_pf = valid_piping_calculator._get_scenario_pf(
            _beta, _mechanism_input_dict
        )

        # 3. Verify expectations.
        _expected_pf = beta_to_pf(_beta)
        assert _computed_pf == pytest.approx(_expected_pf, rel=1e-4)
