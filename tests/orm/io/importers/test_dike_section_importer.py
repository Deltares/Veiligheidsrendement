import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData


class TestDikeSectionImporter:
    def test_initialize(self):
        _importer = DikeSectionImporter()
        assert isinstance(_importer, DikeSectionImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DikeSectionImporter()

        # 2. Run test
        _dike_section = _importer.import_orm(SectionData.get_by_id(1))

        # 3. Verify expectations.
        assert isinstance(_dike_section, DikeSection)
        assert _dike_section.name == "section_one"
        assert isinstance(_dike_section.houses, pd.DataFrame)
        assert isinstance(_dike_section.mechanism_data, dict)
        assert any(_dike_section.mechanism_data.items())

    def test_import_buildings_list(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = DikeSectionImporter()
        _section_data = SectionData.get_by_id(1)
        _buildings_query = Buildings.select().where(
            Buildings.section_data == _section_data
        )

        # 2. Run test
        _buildings_frame = _importer._import_buildings_list(_buildings_query)

        # 3. Verify expectations.
        assert isinstance(_buildings_frame, pd.DataFrame)
        assert list(_buildings_frame.columns) == ["distancefromtoe", "cumulative"]
        assert len(_buildings_frame) == 2
        assert list(_buildings_frame.loc[0]) == [24, 2]
        assert list(_buildings_frame.loc[1]) == [42, 1]

    def test_import_orm_without_model_raises_value(self):
        # 1. Define test data.
        _importer = DikeSectionImporter()
        _expected_mssg = "No valid value given for SectionData."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg
