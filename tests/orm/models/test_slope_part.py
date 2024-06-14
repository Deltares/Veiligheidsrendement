from typing import Callable

import pytest

from tests.orm import with_empty_db_fixture
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.slope_part import SlopePart


class TestSlopePart:
    @with_empty_db_fixture
    def test_initialize_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario = get_basic_computation_scenario()

        # 2. Run test
        _slope_part = SlopePart.create(
            computation_scenario=_scenario,
            begin_part=13.37,
            end_part=37.13,
            top_layer_type=20.1,
            tan_alpha=0.25064,
        )

        # 3. Verify expectations.
        assert isinstance(_slope_part, SlopePart)
        assert isinstance(_slope_part, OrmBaseModel)
        assert _slope_part.computation_scenario == _scenario
        assert _slope_part in _scenario.slope_parts
