import pytest

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject


class TestDikeTraject:
    def test_from_config_without_traject_raises_value_error(
        self,
    ):
        # Setup
        config = VrtoolConfig()

        # Call
        with pytest.raises(ValueError) as value_error:
            DikeTraject.from_config(config)

        # Assert
        assert str(value_error.value) == "No traject given in config."
