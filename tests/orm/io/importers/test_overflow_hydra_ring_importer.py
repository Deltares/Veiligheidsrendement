import pytest

from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.orm.io.overflow_hydra_ring_importer import OverFlowHydraRingImporter
from vrtool.orm.models.computation_scenario import ComputationScenario


class TestOverflowHydraRingImporter:
    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = OverFlowHydraRingImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(ComputationScenario.get_by_id(1))

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 2
        assert _mechanism_input.input["h_crest"] == pytest.approx(9.13)
        assert _mechanism_input.input["d_crest"] == pytest.approx(0.005)
