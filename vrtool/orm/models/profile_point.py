from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from peewee import ForeignKeyField, FloatField
from vrtool.orm.models.characteristic_point_type import CharacteristicPointType
from vrtool.orm.models.section_data import SectionData

class ProfilePoint(OrmBaseModel):
    profile_point_type = ForeignKeyField(CharacteristicPointType, backref="profile_points")
    section_data = ForeignKeyField(SectionData, backref="profile_points")
    x_coordinate = FloatField(null=False)
    y_coordinate = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)