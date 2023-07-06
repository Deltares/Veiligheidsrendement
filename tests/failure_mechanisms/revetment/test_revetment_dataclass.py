import pytest

from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)


class TestRevetmentDataClass:
    def test_current_transition_level(self):
        # 1. Define test data.
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(StoneSlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(StoneSlopePart(2, 3, 0.32, 26.1, 0.15))
        revetments.slope_parts.append(GrassSlopePart(3, 4, 0.33, 20))
        revetments.slope_parts.append(GrassSlopePart(4, 5, 0.34, 20))

        level = revetments.current_transition_level

        assert level == 3

    def test_current_transition_level_no_grass(self):
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(StoneSlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(StoneSlopePart(2, 3, 0.32, 26.1, 0.15))

        with pytest.raises(ValueError) as exception_error:
            revetments.current_transition_level

        # Assert
        assert str(exception_error.value) == "No slope part with grass found."

    def test_available_years(self):
        # 1. Define test data.
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(StoneSlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(StoneSlopePart(2, 3, 0.32, 26.1, 0.15))
        revetments.slope_parts.append(GrassSlopePart(3, 4, 0.33, 20))
        revetments.slope_parts.append(GrassSlopePart(4, 5, 0.34, 20))
        revetments.grass_relations.append(RelationGrassRevetment(2020, 1.0, 2.0))
        for i in range(2):
            revetments.slope_parts[i].slope_part_relations.append(
                RelationStoneRevetment(2020, 2.0, 3.0)
            )

        given_years = revetments.find_given_years()

        assert len(given_years) == 1
        assert given_years[0] == 2020

    def test_available_years_when_relations_are_not_same_years_raises_value_error(self):
        # 1. Define test data.
        revetments = RevetmentDataClass()
        revetments.slope_parts.append(StoneSlopePart(1, 2, 0.31, 5, 0.1))
        revetments.slope_parts.append(StoneSlopePart(2, 3, 0.32, 26.1, 0.15))
        revetments.slope_parts.append(GrassSlopePart(3, 4, 0.33, 20))
        revetments.slope_parts.append(GrassSlopePart(4, 5, 0.34, 20))
        revetments.grass_relations.append(RelationGrassRevetment(2020, 1.0, 2.0))

        with pytest.raises(ValueError) as value_error:
            revetments.find_given_years()

        assert str(value_error.value) == "Years for grass and stone differ."
