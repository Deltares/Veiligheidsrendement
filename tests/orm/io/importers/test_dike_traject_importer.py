from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from tests.orm.io.importers import db_fixture
from peewee import SqliteDatabase
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
import pytest

class TestDikeTrajectInfoImporter:
    def test_initialize(self):
        _importer = DikeTrajectImporter()
        assert isinstance(_importer, DikeTrajectImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DikeTrajectImporter()

        # 2. Run test.
        _dike_traject = _importer.import_orm(OrmDikeTrajectInfo.get_by_id(1))

        # 3. Verify final expectations.
        assert isinstance(_dike_traject, DikeTraject)
        assert isinstance(_dike_traject.general_info, DikeTrajectInfo)
        assert isinstance(_dike_traject.sections, list)
        assert any(_dike_traject.sections)
        assert all(isinstance(_section, DikeSection) for _section in _dike_traject.sections)

    def test_import_orm_without_model_raises_value(self):
        # 1. Define test data.
        _importer = DikeTrajectImporter()
        _expected_mssg = "No valid value given for DikeTrajectInfo."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg