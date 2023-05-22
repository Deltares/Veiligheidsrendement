import pytest

from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.orm.io.dstability_importer import DStabilityImporter
from vrtool.orm.models.computation_scenario import ComputationScenario


class TestDStabilityImporter:
    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DStabilityImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(ComputationScenario.get_by_id(2))

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 1
        assert (
            _mechanism_input.input["stix_file"]
            == "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
