import shutil
import pytest
from peewee import SqliteDatabase

from tests import test_results
from vrtool.orm.orm_controllers import initialize_database
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData

@pytest.fixture(autouse=False)
def empty_db_fixture(request: pytest.FixtureRequest):
    _parts = request.node.nodeid.split("::")
    _db_file = test_results / _parts[-2] / f"{_parts[-1]}_db.db"
    if _db_file.exists():
        _db_file.unlink()

    _db = initialize_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    yield _db

    shutil.rmtree(_db_file.parent)