from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo as VrtoolDikeTrajectInfo
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as DikeTrajectInfo


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
        assert _dike_traject_info.traject_name == "sth"
        assert _dike_traject_info.traject_name==1
        assert _dike_traject_info.omegaPiping==2
        assert _dike_traject_info.omegaStabilityInner==3
        assert _dike_traject_info.omegaOverflow==4
        assert _dike_traject_info.aPiping==5
        assert _dike_traject_info.bPiping==6
        assert _dike_traject_info.aStabilityInner==7
        assert _dike_traject_info.bStabilityInner==8
        assert _dike_traject_info.beta_max==0
        assert _dike_traject_info.Pmax==8
        assert _dike_traject_info.FloodDamage==8
        assert _dike_traject_info.TrajectLength==6
