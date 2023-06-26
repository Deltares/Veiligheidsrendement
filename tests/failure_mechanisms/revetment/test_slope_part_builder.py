from typing import Type
from vrtool.failure_mechanisms.revetment.asphalt_slope_part import AsphaltSlopePart
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part_builder import SlopePartBuilder
import pytest

from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart


class TestSlopePartBuilder:
    @pytest.mark.parametrize(
        "top_layer_type, expected_type",
        [
            pytest.param(20.0, GrassSlopePart, id="Grass slope part type"),
            pytest.param(26.0, StoneSlopePart, id="Stone slope part type (lower)"),
            pytest.param(27.0, StoneSlopePart, id="Stone slope part type"),
            pytest.param(27.9, StoneSlopePart, id="Stone slope part type (upper)"),
            pytest.param(5.0, AsphaltSlopePart, id="Asphalt slope part type"),
        ],
    )
    def test_get_slope_part_type_with_known_values(
        self, top_layer_type: float, expected_type: Type[SlopePartProtocol]
    ):
        _slope_type = SlopePartBuilder.get_slope_part_type(top_layer_type)
        assert _slope_type == expected_type

    def test_get_slope_part_type_with_unknown_value_raises(self):
        with pytest.raises(ValueError) as exc_err:
            SlopePartBuilder.get_slope_part_type(-1)

        assert str(exc_err.value) == "No SlopePart type found for top layer type: -1."
