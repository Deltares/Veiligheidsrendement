import pytest

from vrtool.common.enums import MechanismEnum, VrtoolEnum


class TestVrtoolEnums:
    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("StabilityInner", "STABILITY_INNER", id="VALID CamelCase"),
            pytest.param("OVERFLOW", "OVERFLOW", id="VALID UPPER"),
            pytest.param("Invalid", None, id="INVALID CamelCase"),
        ],
    )
    def test_get_enum(self, enum_name: str, expected: str):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        if _mech:
            assert _mech.name == expected

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("CamelCase", "CAMEL_CASE", id="VALID Normalized CamelCase"),
            pytest.param(None, None, id="INVALID None"),
        ],
    )
    def test_normalize_enum_name(self, enum_name: str, expected: str):
        # 1. Setup

        # 2. Call
        _enum_name = VrtoolEnum._normalize_name(enum_name)

        # 3. Assert
        assert _enum_name == expected

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param(
                "OVERFLOW", "Overflow", id="VALID Denormalized CamelCase (simple)"
            ),
            pytest.param(
                "STABILITY_INNER",
                "StabilityInner",
                id="VALID Denormalized CamelCase (with _)",
            ),
        ],
    )
    def test_denormalize_enum_name(self, enum_name: str, expected: str):
        # 1. Setup
        _enum = MechanismEnum[enum_name]

        # 2. Call
        _enum_name = _enum._denormalize_name(enum_name)

        # 3. Assert
        assert _enum_name == expected
