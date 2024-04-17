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

        given_years = revetments.get_available_years()

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
            revetments.get_available_years()

        assert str(value_error.value) == "Years for grass and stone differ."

    @pytest.mark.parametrize(
        "transition_levels, threshold, expected_result",
        [
            pytest.param([0.2, 0.4, 0.8], 0.4, 0.4, id="Max value equal to threshold"),
            pytest.param([0.2, 0.4, 0.8], 0.5, 0.4, id="Max value less than threshold"),
            pytest.param(
                [0.6, 0.7000000000001],
                0.7,
                0.7000000000001,
                id="Max value 'is close' to threshold",
            ),
        ],
    )
    def test_get_transition_level_below_threshold_returns_transition_level_subset(
        self, transition_levels: list[float], threshold: float, expected_result: float
    ):
        # 1. Define test data.
        _revetment_dc = RevetmentDataClass(
            grass_relations=[
                RelationGrassRevetment(2020, tl, 4.2) for tl in transition_levels
            ]
        )

        # 2. Run test.
        _result = _revetment_dc.get_transition_level_below_threshold(threshold)

        # 3. Verify final expectations.
        assert _result == expected_result

    def test_get_transition_level_below_threshold_when_threshold_is_low_then_raises(
        self,
    ):
        # 1. Define test data.
        _threshold = 0.5
        _revetment_dc = RevetmentDataClass(
            grass_relations=[RelationGrassRevetment(2020, tl, 4.2) for tl in [1, 2, 3]]
        )

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _revetment_dc.get_transition_level_below_threshold(_threshold)

        # 3. Verify final expectations.
        assert str(
            value_error.value
        ) == "No values found below the threshold {}".format(_threshold)

    def test_beta_stone(self):
        """
        test the setter for beta stone
        """

        # 1. Define test data.
        revetments = RevetmentDataClass()
        revetments.set_beta_stone(3.4)

        assert revetments.beta_stone == 3.4
