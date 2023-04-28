from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo as VrtoolDikeTrajectInfo
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as DikeTrajectInfo

import pytest

class TestDikeTrajectInfoImporter:
    def test_initialize(self):
        _importer = DikeTrajectInfoImporter()
        assert isinstance(_importer, DikeTrajectInfoImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DikeTrajectInfoImporter()

        # 2. Run test.
        _dike_traject_info = _importer.import_orm(DikeTrajectInfo.get_by_id(1))

        # 3. Verify final expectations.
        assert isinstance(_dike_traject_info, VrtoolDikeTrajectInfo)
        assert _dike_traject_info.traject_name == "16-1"
        assert _dike_traject_info.omegaPiping == 0.25
        assert _dike_traject_info.omegaStabilityInner == 0.04
        assert _dike_traject_info.omegaOverflow == 0.24
        assert _dike_traject_info.aPiping is None
        assert _dike_traject_info.bPiping == 300
        assert _dike_traject_info.aStabilityInner == 0.033
        assert _dike_traject_info.bStabilityInner == 50
        assert _dike_traject_info.beta_max == pytest.approx(3.7190164854556804)
        assert _dike_traject_info.Pmax == 0.0001
        assert _dike_traject_info.FloodDamage is None
        assert _dike_traject_info.TrajectLength == 0
