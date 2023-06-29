from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)


class TestRelationStoneRevetment:
    def test_relation_stone(self):
        stone = RelationStoneRevetment(2050, 2.3, 3.2)

        assert isinstance(stone, RelationStoneRevetment)
        assert isinstance(stone, RelationRevetmentProtocol)
        assert stone.year == 2050
        assert stone.beta == 3.2
        assert stone.top_layer_thickness == 2.3
