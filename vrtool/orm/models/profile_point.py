from vrtool.orm.models.base_model import BaseModel, _get_table_name
from peewee import ForeignKeyField, FloatField
from vrtool.orm.models.characteristic_point_type import CharacteristicPointType, SectionData

class ProfilePoint(BaseModel):
    profile_point_type = ForeignKeyField(CharacteristicPointType, backref="profile_points")
    section_data = ForeignKeyField(SectionData, backref="profile_points")
    x_coordinate = FloatField(null=False)
    y_coordinate = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)