from typing import Callable

import pytest

from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.slope_part import SlopePart


class TestBlockRevetmentRelation:
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario = get_basic_computation_scenario()
        _slope_part = SlopePart.create(
            computation_scenario=_scenario,
            begin_part=13.37,
            end_part=37.13,
            top_layer_type=20.1,
            tan_alpha=0.25064,
        )

        # 2. Run test
        _block_relation = BlockRevetmentRelation.create(
            slope_part=_slope_part,
            year=2001,
            top_layer_thickness=0.2196,
            beta=0.25064,
        )

        # 3. Verify expectations.
        assert isinstance(_block_relation, BlockRevetmentRelation)
        assert isinstance(_block_relation, OrmBaseModel)
        assert _block_relation.slope_part == _slope_part
        assert _block_relation in _slope_part.block_revetment_relations
