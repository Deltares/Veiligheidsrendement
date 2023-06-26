from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart


class TestStoneSlopePart:
    def test_initialize(self):
        slope = StoneSlopePart(1, 2, 0.333, 26.1, 0.1)

        assert isinstance(slope, StoneSlopePart)
        assert isinstance(slope, SlopePartProtocol)
        assert slope.begin_part == 1
        assert slope.end_part == 2
        assert slope.tan_alpha == 0.333
        assert slope.top_layer_type == 26.1
        assert slope.top_layer_thickness == 0.1
