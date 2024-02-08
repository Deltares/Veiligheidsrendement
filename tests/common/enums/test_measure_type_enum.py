import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum


class TestMeasureTypeEnum:
    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("Soil reinforcement", id="VALID With spacelower"),
            pytest.param("Soil Reinforcement", id="VALID With Spaceupper"),
            pytest.param("SOIL_REINFORCEMENT", id="VALID UPPER_SNAKE"),
            pytest.param("SoilReinforcement", id="VALID CamelCase"),
            pytest.param(" Soil reinforcement", id="VALID Space before"),
            pytest.param("Soil reinforcement ", id="VALID Space after"),
        ],
    )
    def test_get_valid_measure_type_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _measure_type = MeasureTypeEnum.get_enum(enum_name)

        # 3. Assert
        assert _measure_type.name == "SOIL_REINFORCEMENT"

    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("Soilreinforcement", id="INVALID Withoutspace"),
        ],
    )
    def test_get_invalid_measure_type_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _measure_type = MeasureTypeEnum.get_enum(enum_name)

        # 3. Assert
        assert _measure_type.name == "INVALID"

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("SOIL_REINFORCEMENT", "Soil reinforcement", id="VALID SOIL"),
            pytest.param("STABILITY_SCREEN", "Stability Screen", id="VALID STABILITY"),
        ],
    )
    def test_get_valid_old_measure_type_name(self, enum_name: str, expected: str):
        # 1. Setup
        _measure_type = MeasureTypeEnum.get_enum(enum_name)

        # 2. Call
        _measure_type_name = _measure_type.get_old_name()

        # 3. Assert
        assert _measure_type_name == expected
