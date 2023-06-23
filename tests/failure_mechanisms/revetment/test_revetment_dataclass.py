import pytest

from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePart


class TestRevetmentDataClass:
    def test_basic_usage(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(SlopePart(1, 2, 0.31, 5, 0.1))
        revetments.grass_relations.append(RelationGrassRevetment(2030, 2.4, 4.5))
        revetments.block_relations.append(RelationStoneRevetment(0, 2050, 1.2, 3.1))

        assert len(revetments.slope_parts) == 1
        assert len(revetments.grass_relations) == 1
        assert len(revetments.block_relations) == 1

    def test_current_transition_level(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(SlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(SlopePart(2, 3, 0.32, 26.1, 0.15))
        revetments.slope_parts.append(SlopePart(3, 4, 0.33, 20))
        revetments.slope_parts.append(SlopePart(4, 5, 0.34, 20))

        level = revetments.current_transition_level

        assert level == 3

    def test_current_transition_level_no_grass(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(SlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(SlopePart(2, 3, 0.32, 26.1, 0.15))

        with pytest.raises(ValueError) as exception_error:
            revetments.current_transition_level

        # Assert
        assert str(exception_error.value) == "No slope part with grass found"
