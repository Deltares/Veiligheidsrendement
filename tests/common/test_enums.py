import pytest

from vrtool.common.enums import MeasureTypeEnum, MechanismEnum


class TestVrtoolEnum:
    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("Soil reinforcement", id="VALID With space"),
            pytest.param("SOIL_REINFORCEMENT", id="VALID UPPER_SNAKE"),
            pytest.param(" Soil reinforcement", id="VALID space before"),
            pytest.param("Soil reinforcement ", id="VALID space after"),
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
            pytest.param("StabilityInner", id="VALID CamelCase"),
            pytest.param("STABILITY_INNER", id="VALID UPPER_SNAKE"),
            pytest.param("stability_inner", id="VALID lower_snake"),
            pytest.param(" StabilityInner", id="VALID space before"),
            pytest.param("StabilityInner ", id="VALID space after"),
        ],
    )
    def test_get_valid_mechanism_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _mechanism = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        assert _mechanism.name == "STABILITY_INNER"

    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("Soilreinforcement", id="INVALID Withoutspace"),
            pytest.param("SoilReinforcement", id="INVALID CamelCase"),
        ],
    )
    def test_get_invalid_measure_type_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _measure_type = MeasureTypeEnum.get_enum(enum_name)

        # 3. Assert
        assert _measure_type.name == "INVALID"

    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("stabilityinner", id="INVALID camelcase"),
            pytest.param("Stability Inner", id="INVALID space within"),
        ],
    )
    def test_get_invalid_mechanism_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _mechanism = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        assert _mechanism.name == "INVALID"

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("SOIL_REINFORCEMENT", "Soil reinforcement", id="VALID SOIL"),
            pytest.param("STABILITY_SCREEN", "Stability Screen", id="VALID STABILITY"),
        ],
    )
    def test_get_valid_old_measure_tpe_name(self, enum_name: str, expected: str):
        # 1. Setup
        _measure_type = MeasureTypeEnum.get_enum(enum_name)

        # 2. Call
        _measure_type_name = _measure_type.get_old_name()

        # 3. Assert
        assert _measure_type_name == expected

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("OVERFLOW", "Overflow", id="VALID UPPER"),
            pytest.param("STABILITY_INNER", "StabilityInner", id="VALID UPPER_SNAKE"),
        ],
    )
    def test_get_valid_old_mechanism_name(self, enum_name: str, expected: str):
        # 1. Setup
        _mechanism = MechanismEnum.get_enum(enum_name)

        # 2. Call
        _mech_name = _mechanism.get_old_name()

        # 3. Assert
        assert _mech_name == expected
