import pytest

from vrtool.common.enums import MechanismEnum


class TestMechanismEnums:
    @pytest.mark.parametrize(
        "_enum_name, expected",
        [
            pytest.param("CamelCase", "CAMEL_CASE", id="VALID Normalized CamelCase"),
            pytest.param(None, None, id="INVALID None"),
        ],
    )
    def test_normalize_enum_name(self, _enum_name: str, expected: str):
        # 1. Setup

        # 2. Call
        _mech_name = MechanismEnum._normalize_name(_enum_name)

        # 3. Assert
        assert _mech_name == expected

    @pytest.mark.parametrize(
        "_enum_name, expected",
        [
            pytest.param("StabilityInner", "STABILITY_INNER", id="VALID CamelCase"),
            pytest.param("OVERFLOW", "OVERFLOW", id="VALID UPPER"),
            pytest.param("Invalid", None, id="INVALID CamelCase"),
        ],
    )
    def test_get_enum(self, _enum_name: str, expected: str):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum(_enum_name)

        # 3. Assert
        if _mech:
            assert _mech.name == expected
