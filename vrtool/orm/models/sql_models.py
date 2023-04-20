from peewee import IntegerField, TextField, BooleanField, FloatField, ForeignKeyField
from peewee import Model
from vrtool.orm import vrtool_db

def _get_table_name(qual_name: str) -> str:
    """
    When invoking the Meta inner class we can access the `__qual__` attribute which contains its parent class with the name to be used as a SQLite table

    Args:
        qual_name (str): Value of the `__qual__` attribute.

    Returns:
        str: Name of the table.
    """
    return qual_name.split(".")[0]

class BaseModel(Model):
    class Meta:
        database = vrtool_db

class SectionData(BaseModel):
    section_name = TextField(unique=True)
    dijkpaal_start = TextField(null=True)
    dijkpaal_end = TextField(null=True)
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
    name: TextField(unique=True)

    class Meta:
        table_name = _get_table_name(__qualname__)

class MechanismPerSection(BaseModel):
    section = ForeignKeyField(SectionData, backref="mechanisms")
    mechanism = ForeignKeyField(Mechanism, backref="sections")
    class Meta:
        table_name = _get_table_name(__qualname__)

class ComputationType(BaseModel):
    """
    Possible values:
        * Simple
        * HRING
        * SemiProb
    """
    name = TextField(unique=True)
    class Meta:
        table_name = _get_table_name(__qualname__)

class ComputationScenario(BaseModel):
    mechanism_per_section = ForeignKeyField(MechanismPerSection, backref="computation_scenarios")
    computation_type = ForeignKeyField(ComputationType, backref="computation_scenarios")
    computation_name = TextField(null=False)
    scenario_name = TextField()
    scenario_probability = FloatField(null=False)
    probability_of_failure = FloatField()
    class Meta:
        table_name = _get_table_name(__qualname__)

class Parameter(BaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="parameters")
    parameter = TextField(unique=True)
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
    name = TextField(unique=True)

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

    distance_from_toe = TextField(null=False)
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
    name = TextField(unique=True, choices=["SOIL_REINFORCEMENT"])

    class Meta:
        table_name = _get_table_name(__qualname__)

class CombinableType(BaseModel):
    """
    Existing types:
        * full
        * combinable
        * partial
    """
    name = TextField(unique=True)

    class Meta:
        table_name = _get_table_name(__qualname__)

class Measure(BaseModel):
    """
    This should be defined as an abstract class.
    """
    measure_type = ForeignKeyField(MeasureType, backref="measures")
    combinable_type = ForeignKeyField(CombinableType, backref="measures")
    name = TextField()
    year = IntegerField(default=2025)

    class Meta:
        table_name = _get_table_name(__qualname__)

class StandardMeasure(Measure):
    max_inward_reinforcement = IntegerField(default=50)
    max_outward_reinforcement = IntegerField(default=0)
    direction = TextField(default='Inward')
    crest_step = FloatField(default=0.5)
    max_crest_increase = FloatField(default=2)
    stability_screen = BooleanField(default=0)
    prob_of_solution_failure = FloatField(default = 1/1000)
    failure_probability_with_solution = FloatField(default=10**-12)
    stability_screen_s_f_increase = FloatField(default=0.2)

    class Meta:
        table_name = _get_table_name(__qualname__)

class CustomMeasure(Measure):
    mechanism = ForeignKeyField(Mechanism, backref="measures")
    cost = FloatField(null=False)
    beta = FloatField(null=False)
    year = IntegerField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
