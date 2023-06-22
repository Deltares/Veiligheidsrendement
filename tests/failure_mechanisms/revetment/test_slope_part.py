from vrtool.failure_mechanisms.revetment.slope_part import SlopePart


class TestSlopePart:
    def test_slope_part_grass(self):
        slope = SlopePart(1, 2, 0.333, 20)

        assert slope.begin_part == 1
        assert slope.end_part == 2
        assert slope.tan_alpha == 0.333
        assert slope.top_layer_type == 20
        assert slope.is_grass

    def test_slope_part_stone(self):
        slope = SlopePart(1, 2, 0.333, 26.1, 0.1)

        assert slope.begin_part == 1
        assert slope.end_part == 2
        assert slope.tan_alpha == 0.333
        assert slope.top_layer_type == 26.1
        assert slope.top_layer_thickness == 0.1
        assert not slope.is_grass
