from vrtool.orm.models.base_model import BaseModel
from peewee import IntegerField, TextField, BooleanField, FloatField, BlobField, ForeignKeyField

class SectionData(BaseModel):
    section_name: TextField(unique=True)
    dijkpaal_start: TextField()
    dijkpaal_end: TextField()
    meas_start: FloatField()
    meas_end: FloatField()
    section_length: FloatField()
    in_analysis: BooleanField()
    crest_height: FloatField()
    annual_crest_decline: FloatField()
    cover_layer_thickness: FloatField(default=7)
    pleistocene_level: FloatField(default=25)

class Mechanism(BaseModel):
    name: TextField(unique=True)


class CharacteristicPoint(BaseModel):
    name = TextField(unique=True)

class ProfilePoint(BaseModel):
    point_name = TextField(unique=True)
    characteristic_point = ForeignKeyField(CharacteristicPoint, backref='profile_points')
    section_data = ForeignKeyField(SectionData, backref="profile_points")
    x_coordinate = FloatField()
    y_coordinate = FloatField()

class WaterlevelData(BaseModel):
    section_data = ForeignKeyField(SectionData, backref="water_level_data_list")
    water_level_location_id = IntegerField()
    year = IntegerField()
    water_level = FloatField()
    beta = FloatField()

class Buildings(BaseModel):
    section_data = ForeignKeyField(SectionData, backref="buildings_list")

    distance_from_toe = TextField()
    number_of_buildings = IntegerField()

class MeasureType(BaseModel):
    # TODO: This was possible to have as an enum
    name = TextField(primary_key=True)

class CombinableType(BaseModel):
    # TODO: This was possible to have as an enum
    name = TextField(primary_key=True)

class Measure(BaseModel):
    measure_type = ForeignKeyField(MeasureType, backref="measures")
    combinable_type = ForeignKeyField(CombinableType, backref="measures")
    name = TextField(primary_key=True)
    year = IntegerField(default=2025)

class Default(Measure):
    max_inward_reinforcement = IntegerField(default=50)
    max_outward_reinforcement = IntegerField(default=0)
    direction = TextField(default='Inward')
    crest_step = FloatField(default=0.5)
    max_crest_increase = FloatField(default=2)
    stability_screen = BooleanField(default=0)
    prob_of_solution_failure = FloatField(default = 1/1000)
    failure_probability_with_solution = FloatField(default=10**-12)
    stability_screen_s_f_increase = FloatField(default=0.2)

class Custom(Measure):
    mechanism = ForeignKeyField(Mechanism, backref="measures")
    cost = FloatField()
    beta = FloatField()
    year = IntegerField()

