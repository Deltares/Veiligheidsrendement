import pytest
from pytest import approx
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class TestDikeTrajectInfo:
    def test_from_traject_name_16_4_sets_correct_properties(self):
        # Setup
        traject_name = "16-4"

        # Call
        info = DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert info.traject_name == traject_name
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

    def test_from_traject_name_16_3_sets_correct_properties(self):
        # Setup
        traject_name = "16-3"

        # Call
        info = DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert info.traject_name == traject_name
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

    def test_from_traject_name_16_3_and_16_4_sets_correct_properties(self):
        # Setup
        traject_name = "16-3 en 16-4"

        # Call
        info = DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert info.traject_name == traject_name
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

    def test_from_traject_name_38_1_sets_correct_properties(self):
        # Setup
        traject_name = "38-1"

        # Call
        info = DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert info.traject_name == traject_name
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

    def test_from_traject_name_16_1_sets_correct_properties(self):
        # Setup
        traject_name = "16-1"

        # Call
        info = DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert info.traject_name == traject_name
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
    def test_from_traject_name_unknown_traject_sets_default_properties(
        self, traject_name: str
    ):
        # Call
        with pytest.raises(ValueError) as exception_error:
            DikeTrajectInfo.from_traject_name(traject_name)

        # Assert
        assert (
            str(exception_error.value)
            == f"Warning: Traject {traject_name} not recognised."
        )
