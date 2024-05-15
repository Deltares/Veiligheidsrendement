import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum


class TestCombinableTypeEnum:
    @pytest.mark.parametrize(
        "enum_name",
        [
            pytest.param("full", id="VALID lower"),
            pytest.param("FULL", id="VALID UPPER"),
            pytest.param("Full", id="VALID CamelCase"),
            pytest.param(" full", id="VALID space before"),
            pytest.param("full ", id="VALID space after"),
        ],
    )
    def test_get_valid_combinable_type_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _combinable_type = CombinableTypeEnum.get_enum(enum_name)

        # 3. Assert
        assert _combinable_type.name == "FULL"

    @pytest.mark.parametrize(
        "enum_name",
        [],
    )
    def test_get_invalid_combinable_type_enum(self, enum_name: str):
        # 1. Setup

        # 2. Call
        _combinable_type = CombinableTypeEnum.get_enum(enum_name)

        # 3. Assert
        assert _combinable_type.name == "INVALID"

    @pytest.mark.parametrize(
        "enum_name, expected",
        [
            pytest.param("FULL", "full", id="VALID UPPER"),
        ],
    )
    def test_get_valid_old_mechnism_name(self, enum_name: str, expected: str):
        # 1. Setup
        _combinable_type = CombinableTypeEnum.get_enum(enum_name)

        # 2. Call
        _combinable_type_name = _combinable_type.legacy_name

        # 3. Assert
        assert _combinable_type_name == expected
