from typing import Callable, Iterable

import pytest

from tests.orm import with_empty_db_context
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.slope_part import SlopePart


class TestBlockRevetmentRelation:
    @pytest.fixture(name="slope_part_fixture")
    def _get_slope_part_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ) -> Iterable[SlopePart]:
        _scenario = get_basic_computation_scenario()
        yield SlopePart.create(
            computation_scenario=_scenario,
            begin_part=13.37,
            end_part=37.13,
            top_layer_type=20.1,
            tan_alpha=0.25064,
        )

    @with_empty_db_context
    def test_initialize_with_database_fixture(self, slope_part_fixture: SlopePart):
        # 1. Define test data.
        assert isinstance(slope_part_fixture, SlopePart)

        # 2. Run test
        _block_relation = BlockRevetmentRelation.create(
            slope_part=slope_part_fixture,
            year=2001,
            top_layer_thickness=0.2196,
            beta=0.25064,
        )

        # 3. Verify expectations.
        assert isinstance(_block_relation, BlockRevetmentRelation)
        assert isinstance(_block_relation, OrmBaseModel)
        assert _block_relation.slope_part == slope_part_fixture
        assert _block_relation in slope_part_fixture.block_revetment_relations

    @with_empty_db_context
    def test_on_delete_slope_part_cascades(self, slope_part_fixture: SlopePart):
        # 1. Define test data.
        assert isinstance(slope_part_fixture, SlopePart)
        assert not any(BlockRevetmentRelation.select())

        BlockRevetmentRelation.create(
            slope_part=slope_part_fixture,
            year=2001,
            top_layer_thickness=0.2196,
            beta=0.25064,
        )
        assert any(BlockRevetmentRelation.select())

        # 2. Run test.
        slope_part_fixture.delete_instance()

        # 3. Verify expectations.
        assert not any(BlockRevetmentRelation.select())
