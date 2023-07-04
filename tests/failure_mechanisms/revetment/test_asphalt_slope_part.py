from vrtool.failure_mechanisms.revetment.slope_part import (
    AsphaltSlopePart,
    SlopePartProtocol,
)


class TestAsphaltSlopePart:
    def test_initialize(self):
        slope = AsphaltSlopePart(1, 2, 0.333, 20)

        assert isinstance(slope, AsphaltSlopePart)
        assert isinstance(slope, SlopePartProtocol)
        assert slope.begin_part == 1
        assert slope.end_part == 2
        assert slope.tan_alpha == 0.333
        assert slope.top_layer_type == 20
