from peewee import IntegerField, BooleanField, FloatField, ForeignKeyField, CharField
from peewee import Model
from vrtool.orm.orm_db import vrtool_db

def _get_table_name(qual_name: str) -> str:
    """
    When invoking the Meta inner class we can access the `__qual__` attribute which contains its parent class with the name to be used as a SQLite table

    Args:
        qual_name (str): Value of the `__qual__` attribute.

    Returns:
        str: Name of the table.
    """
    return qual_name.split(".")[0]

_max_char_length = 128

class BaseModel(Model):
    class Meta:
        database = vrtool_db


class DikeTrajectInfo(BaseModel):
    traject_name = CharField(max_length=_max_char_length)
    omega_piping = FloatField(default=0.24)
    omega_stability_inner = FloatField(0.04)
    omega_overflow = FloatField(0.24)
    a_piping = FloatField(default=float("nan"), null=True)
    b_piping = FloatField(default=300)
    a_stability_inner = FloatField(default = 0.033)
    b_stability_inner = FloatField(default = 50)
    beta_max = FloatField(default=float("nan"), null=True)
    p_max = FloatField(default=float("nan"), null=True)
    flood_damage = FloatField(default=float("nan"), null=True)
    traject_length = FloatField(default=float("nan"), null=True)

    class Meta:
        table_name = _get_table_name(__qualname__)

class SectionData(BaseModel):
    dike_traject = ForeignKeyField(DikeTrajectInfo, backref="dike_sections")
    section_name = CharField(unique=True, max_length=_max_char_length)
    dijkpaal_start = CharField(null=True, max_length=_max_char_length)
    dijkpaal_end = CharField(null=True, max_length=_max_char_length)
    meas_start = FloatField()
    meas_end = FloatField()
    section_length = FloatField()
    in_analysis = BooleanField()
    crest_height = FloatField()
    annual_crest_decline = FloatField()
    cover_layer_thickness = FloatField(default=7)
    pleistocene_level = FloatField(default=25)

    class Meta:
        table_name = _get_table_name(__qualname__)

class Mechanism(BaseModel):
    name = CharField(max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)

class MechanismPerSection(BaseModel):
    section = ForeignKeyField(SectionData, backref="mechanisms_per_section")
    mechanism = ForeignKeyField(Mechanism, backref="sections_per_mechanism")
    class Meta:
        table_name = _get_table_name(__qualname__)

class ComputationType(BaseModel):
    """
    Possible values:
        * Simple
        * HRING
        * SemiProb
    """
    name = CharField(unique=True, max_length=_max_char_length)
    class Meta:
        table_name = _get_table_name(__qualname__)

class ComputationScenario(BaseModel):
    mechanism_per_section = ForeignKeyField(MechanismPerSection, backref="computation_scenarios")
    computation_type = ForeignKeyField(ComputationType, backref="computation_scenarios")
    computation_name = CharField(null=False, max_length=_max_char_length)
    scenario_name = CharField(max_length=_max_char_length)
    scenario_probability = FloatField(null=False)
    probability_of_failure = FloatField()
    class Meta:
        table_name = _get_table_name(__qualname__)

class Parameter(BaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="parameters")
    parameter = CharField(unique=True, max_length=_max_char_length)
    value = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)

class MechanismTable(BaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="mechanism_tables")
    year = IntegerField(null=False)
    value = FloatField(null=False)
    beta = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)

class CharacteristicPointType(BaseModel):
    """
    Possible values:
        * `BIT`,
        * `BUT`,
        * `BUK`,
        * `BIK`, 
        * (optionals: `EBL`, `BBL`)
    """
    name = CharField(unique=True, max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)

class ProfilePoint(BaseModel):
    profile_point_type = ForeignKeyField(CharacteristicPointType, backref="profile_points")
    section_data = ForeignKeyField(SectionData, backref="profile_points")
    x_coordinate = FloatField(null=False)
    y_coordinate = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)

class WaterlevelData(BaseModel):
    section_data = ForeignKeyField(SectionData, backref="water_level_data_list")
    water_level_location_id = IntegerField(null=False)
    year = IntegerField(null=False)
    water_level = FloatField(null=False)
    beta = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)

class Buildings(BaseModel):
    section_data = ForeignKeyField(SectionData, backref="buildings_list")

    distance_from_toe = CharField(null=False, max_length=_max_char_length)
    number_of_buildings = IntegerField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)

class MeasureType(BaseModel):
    """
    Existing types:
        * Soil reinforcement
        * Stability screen
        * Soil reinforcement with stability screen
        * Vertical Geotextile
        * Diaphragm wall
    """
    name = CharField(unique=True)

    class Meta:
        table_name = _get_table_name(__qualname__)

class CombinableType(BaseModel):
    """
    Existing types:
        * full
        * combinable
        * partial
    """
    name = CharField(unique=True, max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)

class Measure(BaseModel):
    """
    This should be defined as an abstract class.
    """
    measure_type = ForeignKeyField(MeasureType, backref="measures")
    combinable_type = ForeignKeyField(CombinableType, backref="measures")
    name = CharField(max_length=_max_char_length)
    year = IntegerField(default=2025)

    class Meta:
        table_name = _get_table_name(__qualname__)

class StandardMeasure(BaseModel):
    measure = ForeignKeyField(Measure, backref="standard_measure", unique=True)
    max_inward_reinforcement = IntegerField(default=50)
    max_outward_reinforcement = IntegerField(default=0)
    direction = CharField(default='Inward', max_length=_max_char_length)
    crest_step = FloatField(default=0.5)
    max_crest_increase = FloatField(default=2)
    stability_screen = BooleanField(default=0)
    prob_of_solution_failure = FloatField(default = 1/1000)
    failure_probability_with_solution = FloatField(default=10**-12)
    stability_screen_s_f_increase = FloatField(default=0.2)

    class Meta:
        table_name = _get_table_name(__qualname__)

class CustomMeasure(BaseModel):
    measure = ForeignKeyField(Measure, backref="custom_measures", unique=True)
    mechanism = ForeignKeyField(Mechanism, backref="measures")
    cost = FloatField(null=False)
    beta = FloatField(null=False)
    year = IntegerField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
