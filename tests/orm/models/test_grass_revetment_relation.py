from collections.abc import Callable

from tests.orm import with_empty_db_context
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.grass_revetment_relation import GrassRevetmentRelation
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestGrassRevetmentRelation:
    @with_empty_db_context
    def test_initialize_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario = get_basic_computation_scenario()

        # 2. Run test
        _grass_relation = GrassRevetmentRelation.create(
            computation_scenario=_scenario,
            year=2001,
            transition_level=2,
            beta=0.25064,
        )

        # 3. Verify expectations.
        assert isinstance(_grass_relation, GrassRevetmentRelation)
        assert isinstance(_grass_relation, OrmBaseModel)
        assert _grass_relation.computation_scenario == _scenario
        assert _grass_relation in _scenario.grass_revetment_relations

    @with_empty_db_context
    def test_on_delete_computation_scenario_cascades(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario = get_basic_computation_scenario()
        assert not any(GrassRevetmentRelation.select())

        GrassRevetmentRelation.create(
            computation_scenario=_scenario,
            year=2001,
            transition_level=2,
            beta=0.25064,
        )
        assert any(GrassRevetmentRelation.select())

        # 2. Run test.
        _scenario.delete_instance()

        # 3. Verify expectations.
        assert not any(GrassRevetmentRelation.select())
