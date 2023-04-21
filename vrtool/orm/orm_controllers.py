from pathlib import Path
from peewee import SqliteDatabase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_db import vrtool_db
from vrtool.orm import orm_models as orm
from vrtool.orm.orm_converters import import_dike_traject_info, import_dike_section_list
from vrtool.flood_defence_system.dike_traject import DikeTraject

def initialize_database(database_path: Path):
    """
    Generates an empty SQLite database with all the tables requried by the `Vrtool`.

    Args:
        database_path (Path): Location where to save the database.
    """
    if not database_path.exists():
        database_path.parent.mkdir(parents=True)

    vrtool_db.init(database_path)
    vrtool_db.connect()
    vrtool_db.create_tables([
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
        orm.DikeTrajectInfo])

def open_database(database_path: Path) -> SqliteDatabase:
    """
    Initializes and connects the `Vrtool` to its related database.

    Args:
        database_path (Path): Location of the SQLite database.

    Returns:
        SqliteDatabase: Initialized database.
    """
    vrtool_db.init(database_path)
    vrtool_db.connect()
    return vrtool_db

def get_dike_traject(config: VrtoolConfig) -> DikeTraject:
    """
    Returns a dike traject with all the required section data.
    """
    open_database(config.input_database_path)
    _dike_traject_info = import_dike_traject_info(orm.DikeTrajectInfo.get(orm.DikeTrajectInfo.traject_name == config.traject))

    _dike_traject = DikeTraject()
    _dike_traject.general_info = _dike_traject_info

    _dike_traject.mechanism_names = config.mechanisms
    _dike_traject.assessment_plot_years = config.assessment_plot_years
    _dike_traject.flip_traject = config.flip_traject
    _dike_traject.t_0 = config.t_0
    _dike_traject.T = config.T

    # Currently it is assumed that all SectionData present in a db belongs to whatever traject name has been provided.
    _dike_traject.sections = import_dike_section_list(orm.SectionData.select().where(orm.SectionData.in_analysis == True))
    return _dike_traject
