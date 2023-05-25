from pathlib import Path

from peewee import SqliteDatabase

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm import models as orm
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.orm_db import vrtool_db


def initialize_database(database_path: Path) -> SqliteDatabase:
    """
    Generates an empty SQLite database with all the tables requried by the `Vrtool`.

    Args:
        database_path (Path): Location where to save the database.

    Returns:
        SqliteDatabase: The initialized instance of the database.
    """
    if not database_path.parent.exists():
        database_path.parent.mkdir(parents=True)

    vrtool_db.init(database_path)
    vrtool_db.connect()
    vrtool_db.create_tables(
        [
            orm.SectionData,
            orm.Buildings,
            orm.Mechanism,
            orm.MechanismPerSection,
            orm.ComputationType,
            orm.ComputationScenario,
            orm.Parameter,
            orm.MechanismTable,
            orm.CharacteristicPointType,
            orm.ProfilePoint,
            orm.WaterlevelData,
            orm.MeasureType,
            orm.CombinableType,
            orm.Measure,
            orm.StandardMeasure,
            orm.CustomMeasure,
            orm.DikeTrajectInfo,
            orm.SupportingFile,
            orm.MeasurePerSection,
        ]
    )
    return vrtool_db


def open_database(database_path: Path) -> SqliteDatabase:
    """
    Initializes and connects the `Vrtool` to its related database.

    Args:
        database_path (Path): Location of the SQLite database.

    Returns:
        SqliteDatabase: Initialized database.
    """
    if not database_path.exists():
        raise ValueError("No file was found at {}".format(database_path))
    vrtool_db.init(database_path)
    vrtool_db.connect()
    return vrtool_db


def get_dike_traject(config: VrtoolConfig) -> DikeTraject:
    """
    Returns a dike traject with all the required section data.
    """
    open_database(config.input_database_path)
    _dike_traject = DikeTrajectImporter(config).import_orm(
        orm.DikeTrajectInfo.get(orm.DikeTrajectInfo.traject_name == config.traject)
    )
    return _dike_traject
