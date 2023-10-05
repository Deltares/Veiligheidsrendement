import pytest

from vrtool.common.enums import MechanismEnum


class TestEnums:
    def test_get_enum(self):
        # 1. Setup

        # 2. Call
        _mech = MechanismEnum.get_enum("StabilityInner")

        # 3. Assert
        assert _mech.name == MechanismEnum.STABILITY_INNER.name
        assert _mech.value == MechanismEnum.STABILITY_INNER.value
