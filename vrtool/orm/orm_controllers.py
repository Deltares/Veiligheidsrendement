from pathlib import Path
from peewee import SqliteDatabase
from vrtool.orm.orm_db import vrtool_db
from vrtool.orm.orm_models import SectionData, Buildings, Mechanism, MechanismPerSection, ComputationType, ComputationScenario, Parameter, MechanismTable, CharacteristicPointType, ProfilePoint, WaterlevelData, MeasureType, CombinableType, Measure, StandardMeasure, CustomMeasure, DikeTrajectInfo

def initialize_database(database_path: Path):
    if not database_path.exists():
        database_path.parent.mkdir(parents=True)

    vrtool_db.init(database_path)
    vrtool_db.connect()
    vrtool_db.create_tables([SectionData, Buildings, Mechanism, MechanismPerSection, ComputationType, ComputationScenario, Parameter, MechanismTable, CharacteristicPointType, ProfilePoint, WaterlevelData, MeasureType, CombinableType, Measure, StandardMeasure, CustomMeasure, DikeTrajectInfo])

def open_database(database_path: Path) -> SqliteDatabase:
    vrtool_db.init(database_path)
    vrtool_db.connect()
    return vrtool_db