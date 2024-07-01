import numpy as np
import pytest
from pytest import approx

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.failure_mechanisms.piping.piping_failure_submechanism import (
    PipingFailureSubmechanism,
)
from vrtool.failure_mechanisms.piping.piping_probabilistic_helper import (
    PipingProbabilisticHelper,
)


class TestPipingProbabilisticHelper:
    _traject_name = "16-1"
    _traject_length = 15000
    _probabilistic_functions: PipingProbabilisticHelper = None

    @pytest.fixture(name="piping_probabilistic_helper_setup", autouse=True)
    def _piping_probabilistic_helper_setup_fixture(self):
        _dike_traject_info = DikeTrajectInfo.from_traject_info(
            self._traject_name, self._traject_length
        )
        self._probabilistic_functions = PipingProbabilisticHelper(_dike_traject_info)

    def test_init_with_invalid_dike_traject_info(self):
        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingProbabilisticHelper("NotDikeTrajectInfo")

        # Assert
        assert str(exception_error.value) == "Expected instance of a DikeTrajectInfo."

    @pytest.mark.parametrize(
        "submechanism, expected_gamma",
        [
            pytest.param(PipingFailureSubmechanism.PIPING, 1.1624370714852628),
            pytest.param(PipingFailureSubmechanism.HEAVE, 1.1952560367631886),
            pytest.param(PipingFailureSubmechanism.UPLIFT, 1.5833969960581846),
        ],
    )
    def test_calculate_gamma_supported_mechanism_returns_expected_gamma(
        self, submechanism: PipingFailureSubmechanism, expected_gamma: float
    ):
        # Call
        beta = self._probabilistic_functions.calculate_gamma(submechanism)

        # Assert
        assert beta == approx(expected_gamma)

    @pytest.mark.parametrize(
        "submechanism",
        [
            (PipingFailureSubmechanism.PIPING),
            (PipingFailureSubmechanism.UPLIFT),
            (PipingFailureSubmechanism.HEAVE),
        ],
    )
    def test_calculate_implicated_beta_safety_factor_0_returns_expected_beta(
        self, submechanism: PipingFailureSubmechanism
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(submechanism, 0)

        # Assert
        assert beta == 0.5

    @pytest.mark.parametrize(
        "submechanism",
        [
            (PipingFailureSubmechanism.PIPING),
            (PipingFailureSubmechanism.UPLIFT),
            (PipingFailureSubmechanism.HEAVE),
        ],
    )
    def test_calculate_implicated_beta_safety_factor_infinite_returns_expected_beta(
        self, submechanism: PipingFailureSubmechanism
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(
            submechanism, np.inf
        )

        # Assert
        assert beta == 8

    @pytest.mark.parametrize(
        "safety_factor, expected_reliability",
        [
            (0.01, -7.917845828109184),
            (0.1, -1.6946428740712232),
            (0.5, 2.655189321696615),
            (1, 4.528560079966738),
            (1.5, 5.624411723502317),
            (100, 16.974965988042662),
        ],
    )
    def test_calculate_implicated_beta_piping_returns_expected_beta(
        self,
        safety_factor: float,
        expected_reliability: float,
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(
            PipingFailureSubmechanism.PIPING, safety_factor
        )

        # Assert
        assert beta == approx(expected_reliability)

    @pytest.mark.parametrize(
        "safety_factor, expected_reliability",
        [
            (0.01, -6.074942821791353),
            (0.1, -1.0693230544129941),
            (0.5, 2.4294550160959196),
            (1, 3.936296712965366),
            (1.5, 4.817742600157028),
            (100, 13.947536247722086),
        ],
    )
    def test_calculate_implicated_beta_uplift_returns_expected_beta(
        self,
        safety_factor: float,
        expected_reliability: float,
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(
            PipingFailureSubmechanism.UPLIFT, safety_factor
        )

        # Assert
        assert beta == approx(expected_reliability)

    @pytest.mark.parametrize(
        "safety_factor, expected_reliability",
        [
            (0.01, -5.03032131596281),
            (0.1, -0.23326903889188216),
            (0.5, 3.1197266120124936),
            (1, 4.563783238179046),
            (1.5, 5.40850221340439),
            (100, 14.157887792320903),
        ],
    )
    def test_calculate_implicated_beta_heave_returns_expected_beta(
        self,
        safety_factor: float,
        expected_reliability: float,
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(
            PipingFailureSubmechanism.HEAVE, safety_factor
        )

        # Assert
        assert beta == approx(expected_reliability)
