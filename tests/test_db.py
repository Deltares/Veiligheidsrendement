import shutil
from vrtool.orm import vrtool_db
from tests import test_results
import pytest
from vrtool.orm.models.sql_models import *

def test_initialize_db(request: pytest.FixtureRequest):
    _db_file = test_results / request.node.name / "vrtool_db.db"
    if _db_file.parent.exists():
        shutil.rmtree(_db_file.parent)

    _db_file.parent.mkdir(parents=True)

    vrtool_db.init(_db_file)
    vrtool_db.connect()
    vrtool_db.create_tables([SectionData, Buildings, Mechanism, MechanismPerSection, ComputationType, ComputationScenario, Parameter, MechanismTable, CharacteristicPointType, ProfilePoint, WaterlevelData, MeasureType, CombinableType, Measure, StandardMeasure, CustomMeasure])

    assert _db_file.exists()