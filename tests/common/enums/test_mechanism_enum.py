import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum


class TestMechanismEnums:
    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("StabilityInner", id="VALID CamelCase"),
            pytest.param("STABILITY_INNER", id="VALID UPPER_SNAKE"),
            pytest.param("stability_inner", id="VALID lower_snake"),
            pytest.param(" StabilityInner", id="VALID space before"),
            pytest.param("StabilityInner ", id="VALID space after"),
            pytest.param("Stability inner ", id="VALID space in between"),
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
            pytest.param("stabilityinner", id="INVALID camelcase"),
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
            pytest.param("OVERFLOW", "Overflow", id="VALID UPPER"),
            pytest.param("STABILITY_INNER", "StabilityInner", id="VALID UPPER_SNAKE"),
        ],
    )
    def test_get_valid_old_mechnism_name(self, enum_name: str, expected: str):
        # 1. Setup
        _mechanism = MechanismEnum.get_enum(enum_name)

        # 2. Call
        _mechanism_name = _mechanism.get_old_name()

        # 3. Assert
        assert _mechanism_name == expected
