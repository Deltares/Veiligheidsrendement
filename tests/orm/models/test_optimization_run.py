import pytest
from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from tests.orm import empty_db_fixture
from peewee import SqliteDatabase, IntegrityError

from vrtool.orm.models.optimization.optimization_type import OptimizationType


class TestOptimizationRun:
    def test_unique_constraint_on_name(self, empty_db_fixture: SqliteDatabase):
        # 1. Define test data.
        _run_name = "DummyRun"
        _opt_type = OptimizationType.create(name="DummyType")
        _opt_run = OptimizationRun.create(
            name=_run_name, discount_rate=0.42, optimization_type=_opt_type
        )
        # 2. Run test
        with pytest.raises(IntegrityError) as db_error:
            OptimizationRun.create(
                name=_opt_run.name, discount_rate=0.42, optimization_type=_opt_type
            )

        # 3. Verify expectations.
        assert str(db_error.value) == "UNIQUE constraint failed: OptimizationRun.name"
