from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)


class TestRelationStoneRevetment:
    def test_relation_stone(self):
        stone = RelationStoneRevetment(0, 2050, 2.3, 3.2)

        assert stone.year == 2050
        assert stone.slope_part == 0
        assert stone.beta == 3.2
        assert stone.top_layer_thickness == 2.3
