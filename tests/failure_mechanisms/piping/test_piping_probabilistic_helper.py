import pytest
from pytest import approx

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.failure_mechanisms.piping.piping_probabilistic_helper import (
    PipingProbabilisticHelper,
)


@pytest.fixture
def configured_functions():
    traject_name = "16-1"
    traject_length = 15000

    info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)
    return PipingProbabilisticHelper(info)


class TestPipingProbabilisticHelper:
    @pytest.fixture(autouse=True)
    def setup(self, configured_functions: PipingProbabilisticHelper):
        self._probabilistic_functions = configured_functions

    @pytest.mark.parametrize(
        "mechanism_name",
        [
            pytest.param(None, id="None mechanism name"),
            pytest.param("   ", id="Whitespace mechanism name"),
            pytest.param("", id="Empty mechanism name"),
            pytest.param("akdjsakld", id="Any mechanism name"),
        ],
    )
    def test_calculate_gamma_unknown_mechanism_raises_error(self, mechanism_name: str):
        # Call
        with pytest.raises(ValueError) as value_error:
            self._probabilistic_functions.calculate_gamma(mechanism_name)

        # Assert
        assert (
            str(value_error.value) == f'Mechanism "{mechanism_name}" is not supported.'
        )

    @pytest.mark.parametrize(
        "mechanism_name, expected_gamma",
        [
            pytest.param("Piping", 1.1624370714852628),
            pytest.param("Heave", 1.1952560367631886),
            pytest.param("Uplift", 1.5833969960581846),
        ],
    )
    def test_calculate_gamma_supported_mechanism_returns_expected_gamma(
        self, mechanism_name: str, expected_gamma: float
    ):
        # Call
        beta = self._probabilistic_functions.calculate_gamma(mechanism_name)

        # Assert
        assert beta == approx(expected_gamma)

    @pytest.mark.parametrize(
        "mechanism_name",
        [
            pytest.param(None, id="None mechanism name"),
            pytest.param("   ", id="Whitespace mechanism name"),
            pytest.param("", id="Empty mechanism name"),
            pytest.param("akdjsakld", id="Any mechanism name"),
        ],
    )
    def test_calculate_implicated_beta_unknown_mechanism_raises_error(
        self, mechanism_name: str
    ):
        # Call
        with pytest.raises(ValueError) as value_error:
            self._probabilistic_functions.calculate_implicated_beta(mechanism_name, 0)

        # Assert
        assert (
            str(value_error.value) == f'Mechanism "{mechanism_name}" is not supported.'
        )

    @pytest.mark.parametrize(
        "mechanism",
        [("Piping"), ("Uplift"), ("Heave")],
    )
    def test_calculate_implicated_beta_safety_factor_0_returns_expected_beta(
        self, mechanism: str
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(mechanism, 0)

        # Assert
        assert beta == 0.5

    @pytest.mark.parametrize(
        "mechanism",
        [("Piping"), ("Uplift"), ("Heave")],
    )
    def test_calculate_implicated_beta_safety_factor_infinite_returns_expected_beta(
        self, mechanism: str
    ):
        # Call
        beta = self._probabilistic_functions.calculate_implicated_beta(
            mechanism, np.inf
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
            "Piping", safety_factor
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
            "Uplift", safety_factor
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
            "Heave", safety_factor
        )

        # Assert
        assert beta == approx(expected_reliability)
