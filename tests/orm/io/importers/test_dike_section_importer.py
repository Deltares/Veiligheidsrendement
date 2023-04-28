from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.models.section_data import SectionData
from tests.orm.io.importers import db_fixture
from peewee import SqliteDatabase
import pandas as pd
from vrtool.flood_defence_system.dike_section import DikeSection

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

