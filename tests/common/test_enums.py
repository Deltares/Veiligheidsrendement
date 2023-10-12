import pytest

from vrtool.common.enums import MechanismEnum


class TestMechanismEnums:
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
    def test_get_valid_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        assert _mech.name == "STABILITY_INNER"

    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("stabilityinner", id="INVALID camelcase"),
            pytest.param("stability inner", id="INVALID space within"),
        ],
    )
    def test_get_invalid_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        assert _mech.name == "INVALID"

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("OVERFLOW", "Overflow", id="VALID UPPER"),
            pytest.param("STABILITY_INNER", "StabilityInner", id="VALID UPPER_SNAKE"),
        ],
    )
    def test_get_valid_old_name(self, enum_name: str, expected: str):
        # 1. Setup
        _mechanism = MechanismEnum.get_enum(enum_name)

        # 2. Call
        _mech_name = _mechanism.get_old_name()

        # 3. Assert
        assert _mech_name == expected
