import shutil
import pytest
from peewee import SqliteDatabase

from tests import test_results, test_data
from vrtool.orm.orm_controllers import initialize_database, open_database
from vrtool.orm.orm_models import *

class TestOrmControllers:

    def test_initialize_db(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _db_file = test_results / request.node.name / "vrtool_db.db"
        if _db_file.parent.exists():
            shutil.rmtree(_db_file.parent)

        # 2. Run test.
        initialize_database(_db_file)

        # 3. Verify expectations.
        assert _db_file.exists()

    def test_open_database(self): 
        # 1. Define test data.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        assert _db_file.is_file()

        # 2. Run test.
        _sql_db = open_database(_db_file)

        # 3. Verify expectations
        assert isinstance(_sql_db, SqliteDatabase)
        assert any(SectionData.select())
        _section_data: SectionData = SectionData.get_by_id(1)
        assert _section_data.section_name == "test_section"
        assert _section_data.dijkpaal_start == "start_point"
        assert _section_data.dijkpaal_end == "end_point"
        assert _section_data.meas_start == 2.4
        assert _section_data.meas_end == 4.2
        assert _section_data.section_length == 123
        assert not _section_data.in_analysis
        assert _section_data.crest_height == 1.0
        assert _section_data.annual_crest_decline == 2.0
        assert _section_data.cover_layer_thickness == 3.0
        assert _section_data.pleistocene_level == 4.0