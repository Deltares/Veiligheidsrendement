from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)


class TestRelationGrassRevetment:
    def test_relation_grass(self):
        grass = RelationGrassRevetment(2025, 2.0, 2.1)

        assert grass.year == 2025
        assert grass.transition_level == 2.0
        assert grass.beta == 2.1
