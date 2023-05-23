import pytest
from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism as OrmMechanism


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
        assert all(
            isinstance(_section, DikeSection) for _section in _dike_traject.sections
        )
        assert _dike_traject.mechanism_names == [
            "a_mechanism",
            "b_mechanism",
            "Overflow",
        ]

    def test_import_orm_without_model_raises_value_error(self):
        # 1. Define test data.
        _importer = DikeTrajectImporter()
        _expected_mssg = "No valid value given for DikeTrajectInfo."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg

    def test_select_available_mechanisms(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DikeTrajectImporter()

        # 2. Run test.
        _mechanisms = _importer._select_available_mechanisms(
            OrmDikeTrajectInfo.get_by_id(1)
        )

        # 3. Verify expectations.
        assert isinstance(_mechanisms, list)
        assert len(_mechanisms) == 3
        assert all(isinstance(_m, OrmMechanism) for _m in _mechanisms)
