import pytest

from pytest import approx
from vrtool.common.dike_traject_info import DikeTrajectInfo


class TestDikeTrajectInfo:
    @pytest.mark.parametrize(
        "traject_name",
        [
            pytest.param("16-3", id="None traject name"),
            pytest.param("16-4", id="Whitespace traject name"),
            pytest.param("16-3 en 16-4", id="Empty traject name"),
        ],
    )
    def test_from_traject_info_16_3_or_16_4_sets_correct_properties(
        self, traject_name: str
    ):
        # Setup
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

    @pytest.mark.parametrize(
        "traject_name",
        [
            pytest.param(None, id="None traject name"),
            pytest.param("   ", id="Whitespace traject name"),
            pytest.param("", id="Empty traject name"),
            pytest.param("akdjsakld", id="Any traject name"),
        ],
    )
    def test_from_traject_info_unknown_traject_raises_error(self, traject_name: str):
        # Setup
        traject_length = 15000

        # Call
        with pytest.raises(ValueError) as value_error:
            DikeTrajectInfo.from_traject_info(traject_name, traject_length)

        # Assert
        assert str(value_error.value) == f"Traject {traject_name} is not supported."
