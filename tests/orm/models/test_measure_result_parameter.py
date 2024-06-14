from typing import Callable

from tests.orm import with_empty_db_fixture
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter,
)
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestMeasureResultParameter:
    @with_empty_db_fixture
    def test_initialize_with_database_fixture(
        self, get_basic_measure_result: Callable[[], MeasureResult]
    ):
        # 1. Define test data.
        _measure_result = get_basic_measure_result()

        # 2. Run test
        _measure_result_parameter = MeasureResultParameter.create(
            name="dberm",
            value=5,
            measure_result=_measure_result,
        )

        # 3. Verify expectations.
        assert isinstance(_measure_result_parameter, MeasureResultParameter)
        assert isinstance(_measure_result_parameter, OrmBaseModel)
        assert _measure_result_parameter.measure_result == _measure_result
        assert _measure_result_parameter in _measure_result.measure_result_parameters
