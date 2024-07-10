from typing import Callable, Iterator

import pytest

from tests.orm import with_empty_db_context
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.mechanism_table import MechanismTable


class TestMechanismTable:
    @pytest.fixture(name="mechanism_table_fixture")
    def _get_computation_scenario_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ) -> Iterator[MechanismTable]:
        _computation_scenario = get_basic_computation_scenario()
        yield MechanismTable.create(
            computation_scenario=_computation_scenario, year=2021, value=23, beta=12
        )

    @with_empty_db_context
    def test_on_delete_section_data_cascades(
        self, mechanism_table_fixture: MechanismTable
    ):
        # 1. Define test data.
        assert isinstance(mechanism_table_fixture, MechanismTable)
        assert any(MechanismTable.select())

        # 2. Run test.
        ComputationScenario.delete_by_id(
            mechanism_table_fixture.computation_scenario.get_id()
        )

        # 3. Verify expectations.
        assert not any(MechanismTable.select())
