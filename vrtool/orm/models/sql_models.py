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
    cover_layer_thickness: FloatField()
    pleistocene_level: FloatField()

class CharacteristicPoint(BaseModel):
    name = TextField(unique=True)

class ProfilePoints(BaseModel):
    point_name = TextField(unique=True)
    characteristic_point = ForeignKeyField(CharacteristicPoint, backref='profile_points_list')
    section_data = ForeignKeyField(SectionData, backref="profile_points_list")
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

