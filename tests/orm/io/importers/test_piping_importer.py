import pytest

from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.models.computation_scenario import ComputationScenario


class TestDStabilityImporter:
    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = PipingImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(ComputationScenario.get_by_id(61))

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 1
