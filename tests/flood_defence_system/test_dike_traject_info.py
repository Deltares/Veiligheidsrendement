import numpy as np
import math
import pytest

from pytest import approx
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class TestDikeTrajectInfo:
    def test_from_traject_info_16_4_sets_correct_properties(self):
        # Setup
        traject_name = "16-4"
        traject_length = 19480

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.9
        assert info.FloodDamage == 23e9
        assert info.TrajectLength == 19480
        assert info.Pmax == 1.0 / 10000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.7190164854556804)
        assert info.gammaHeave == approx(1.292463569010036)
        assert info.gammaPiping == approx(1.3024230063783075)
        assert info.gammaUplift == approx(1.698591299284156)

    def test_from_traject_info_16_3_sets_correct_properties(self):
        # Setup
        traject_name = "16-3"
        traject_length = 19899

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.9
        assert info.FloodDamage == 23e9
        assert info.TrajectLength == 19899
        assert info.Pmax == 1.0 / 10000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.7190164854556804)
        assert info.gammaHeave == approx(1.2950442241929623)
        assert info.gammaPiping == approx(1.3044271319230847)
        assert info.gammaUplift == approx(1.7018414171067793)

    def test_from_traject_info_16_3_and_16_4_sets_correct_properties(self):
        # Setup
        traject_name = "16-3 en 16-4"
        traject_length = 19500

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.9
        assert info.FloodDamage == 23e9
        assert info.TrajectLength == 19500
        assert info.Pmax == 1.0 / 10000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.7190164854556804)
        assert info.gammaHeave == approx(1.2925879351795069)
        assert info.gammaPiping == approx(1.3025196096211376)
        assert info.gammaUplift == approx(1.698747934195857)

    def test_from_traject_info_38_1_sets_correct_properties(self):
        # Setup
        traject_name = "38-1"
        traject_length = 29500

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.9
        assert info.FloodDamage == 14e9
        assert info.TrajectLength == 29500
        assert info.Pmax == 1.0 / 30000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.9878789366069176)
        assert info.gammaHeave == approx(1.3689728103887868)
        assert info.gammaPiping == approx(1.2906177934255822)
        assert info.gammaUplift == approx(1.8033005503044766)

    def test_from_traject_info_16_1_sets_correct_properties(self):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.4
        assert info.FloodDamage == 29e9
        assert info.TrajectLength == 15000
        assert info.Pmax == 1.0 / 30000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.9878789366069176)
        assert info.gammaHeave == approx(1.1952560367631886)
        assert info.gammaPiping == approx(1.1624370714852628)
        assert info.gammaUplift == approx(1.5833969960581846)

    @pytest.mark.parametrize(
        "traject_name",
        [
            pytest.param(None, id="None traject name"),
            pytest.param("   ", id="Whitespace traject name"),
            pytest.param("", id="Empty traject name"),
            pytest.param("akdjsakld", id="Any traject name"),
        ],
    )
    def test_from_traject_info_unknown_traject_sets_default_properties(
        self, traject_name: str
    ):
        # Setup
        traject_length = 15000

        # Call
        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert info.traject_name == traject_name
        assert info.TrajectLength == traject_length

        assert info.aPiping == 0.9
        assert info.FloodDamage == 5e9
        assert info.Pmax == 1.0 / 10000

        assert info.omegaPiping == 0.24
        assert info.omegaStabilityInner == 0.04
        assert info.omegaOverflow == 0.24

        assert info.bPiping == 300

        assert info.aStabilityInner == 0.033
        assert info.bStabilityInner == 50

        assert info.beta_max == approx(3.7190164854556804)
        assert info.gammaHeave == approx(1.2610262138090147)
        assert info.gammaPiping == approx(1.2779345640411361)
        assert info.gammaUplift == approx(1.658976716933591)

    @pytest.mark.parametrize(
        "mechanism",
        [
            pytest.param(None, id="None mechanism name"),
            pytest.param("   ", id="Whitespace mechanism name"),
            pytest.param("", id="Empty mechanism name"),
            pytest.param("akdjsakld", id="Any mechanism name"),
        ],
    )
    def test_calculate_implicated_beta_unknown_mechanism_returns_expected_beta(
        self, mechanism: str
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta(mechanism, 0)

        # Assert
        assert math.isnan(beta)

    @pytest.mark.parametrize(
        "mechanism",
        [("Piping"), ("Uplift"), ("Heave")],
    )
    def test_calculate_implicated_beta_safety_factor_0_returns_expected_beta(
        self, mechanism: str
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta(mechanism, 0)

        # Assert
        assert beta == 0.5

    @pytest.mark.parametrize(
        "mechanism",
        [("Piping"), ("Uplift"), ("Heave")],
    )
    def test_calculate_implicated_beta_safety_factor_infinite_returns_expected_beta(
        self, mechanism: str
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta(mechanism, np.inf)

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
        self, safety_factor: float, expected_reliability: float
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta("Piping", safety_factor)

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
        self, safety_factor: float, expected_reliability: float
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta("Uplift", safety_factor)

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
        self, safety_factor: float, expected_reliability: float
    ):
        # Setup
        traject_name = "16-1"
        traject_length = 15000

        info = DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Call
        beta = info.calculate_implicated_beta("Heave", safety_factor)

        # Assert
        assert beta == approx(expected_reliability)
