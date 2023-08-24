from tests.orm import empty_db_fixture, get_basic_measure_per_section
from vrtool.orm.models import MeasureResult
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestMeasureResult:
    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()

        # 2. Run test
        _measure_result = MeasureResult.create(
            beta=3.1234,
            time=0.0,
            cost=100,
            measure_per_section=_measure_per_section,
        )

        # 3. Verify expectations.
        assert isinstance(_measure_result, MeasureResult)
        assert isinstance(_measure_result, OrmBaseModel)
        assert _measure_result.measure_per_section == _measure_per_section
        assert _measure_result in _measure_per_section.measure_results