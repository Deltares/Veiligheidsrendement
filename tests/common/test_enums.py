import pytest

from vrtool.common.enums import MechanismEnum


class TestMechanismEnums:
    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("StabilityInner", id="VALID CamelCase"),
            pytest.param("STABILITY_INNER", id="VALID UPPER_SNAKE"),
            pytest.param("stability_inner", id="VALID lower_snake"),
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
            pytest.param(" StabilityInner", id="INVALID space before"),
            pytest.param("StabilityInner ", id="INVALID space after"),
        ],
    )
    def test_get_invalid_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum(enum_name)

        # 3. Assert
        assert not _mech
