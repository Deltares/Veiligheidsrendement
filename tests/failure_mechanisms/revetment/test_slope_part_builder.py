import math
from typing import Type
from vrtool.failure_mechanisms.revetment.asphalt_slope_part import AsphaltSlopePart
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part_builder import SlopePartBuilder
import pytest

from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart

_slope_part_cases = [
    pytest.param(20.0, GrassSlopePart, id="Grass slope part type"),
    pytest.param(26.0, StoneSlopePart, id="Stone slope part type (lower)"),
    pytest.param(27.0, StoneSlopePart, id="Stone slope part type"),
    pytest.param(27.9, StoneSlopePart, id="Stone slope part type (upper)"),
    pytest.param(5.0, AsphaltSlopePart, id="Asphalt slope part type"),
]


class TestSlopePartBuilder:
    @pytest.mark.parametrize(
        "top_layer_type, expected_type",
        _slope_part_cases,
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

    @pytest.mark.parametrize("top_layer_type, expected_type", _slope_part_cases)
    def test_build_with_arguments_returns_object(
        self, top_layer_type: float, expected_type: Type[SlopePartProtocol]
    ):
        # 1. Define test data.
        _input_args = {
            "begin_part": 2.4,
            "end_part": 4.2,
            "tan_alpha": 42,
            "top_layer_type": top_layer_type,
        }

        # 2. Run test.
        _slope = SlopePartBuilder.build(**_input_args)

        # 3. Verify expectations.
        assert isinstance(_slope, expected_type)
        assert _slope.begin_part == 2.4
        assert _slope.end_part == 4.2
        assert _slope.tan_alpha == 42
        assert _slope.top_layer_type == top_layer_type
        assert math.isnan(_slope.top_layer_thickness)

    def test_build_with_missing_top_layer_type_raises_error(self):
        with pytest.raises(ValueError) as exc_err:
            SlopePartBuilder.build(top_layer_type=-1)

        assert str(exc_err.value) == "No SlopePart type found for top layer type: -1."
