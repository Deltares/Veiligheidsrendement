import pytest

from vrtool.failure_mechanisms.revetment.revetment_data_class import (
    SlopePart,
    RelationGrassRevetment,
    RelationStoneRevetment,
    RevetmentDataClass,
)


class TestRevetmentDataClass:
    def test_slope_part(self):
        slope = SlopePart(1, 2, 0.333, 20)
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(slope)

        assert len(revetments.slope_parts) == 1
        assert revetments.slope_parts[0].begin_part == 1
        assert revetments.slope_parts[0].end_part == 2
        assert revetments.slope_parts[0].tan_alpha == 0.333
        assert revetments.slope_parts[0].top_layer_type == 20

    def test_relation_grass(self):
        grass = RelationGrassRevetment(2025, 2.0, 2.1)
        revetments = RevetmentDataClass()
        revetments.grass_relations.append(grass)

        assert len(revetments.grass_relations) == 1
        assert revetments.grass_relations[0].year == 2025
        assert revetments.grass_relations[0].transition_level == 2.0
        assert revetments.grass_relations[0].beta == 2.1

    def test_relation_stone(self):
        stone1 = RelationStoneRevetment(0, 2050, 2.3, 3.2)
        stone2 = RelationStoneRevetment(1, 2050, 3.2, 2.3)
        revetments = RevetmentDataClass()
        revetments.stone_relations.append(stone1)
        revetments.stone_relations.append(stone2)

        assert len(revetments.stone_relations) == 2
        assert revetments.stone_relations[0].year == 2050
        assert revetments.stone_relations[1].slope_part == 1

    def test_current_transition_level(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(SlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(SlopePart(2, 3, 0.32, 26.1, 0.15))
        revetments.slope_parts.append(SlopePart(3, 4, 0.33, 20))
        revetments.slope_parts.append(SlopePart(4, 5, 0.34, 20))

        level = revetments.current_transition_level()

        assert level == 3

    def test_current_transition_level_no_grass(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(SlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(SlopePart(2, 3, 0.32, 26.1, 0.15))

        with pytest.raises(ValueError) as exception_error:
            revetments.current_transition_level()

        # Assert
        assert str(exception_error.value) == "No slope part with grass found"
