import pytest
from peewee import SqliteDatabase

from vrtool.orm.io.exporters.measures.custom_measure_time_beta_calculator import (
    CustomMeasureTimeBetaCalculator,
)


# Use fixture from `tests.orm.io.exporters.measures.conftest`
@pytest.mark.usefixtures("custom_measure_db_context")
class TestCustomMeasureTimeBetaCalculator:
    def test_initialize(self, custom_measure_db_context: SqliteDatabase):
        pass
