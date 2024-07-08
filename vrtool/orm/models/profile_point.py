from peewee import FloatField, ForeignKeyField

from vrtool.orm.models.characteristic_point_type import CharacteristicPointType
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class ProfilePoint(OrmBaseModel):
    profile_point_type = ForeignKeyField(
        CharacteristicPointType, backref="profile_points", on_delete="CASCADE"
    )
    section_data = ForeignKeyField(
        SectionData, backref="profile_points", on_delete="CASCADE"
    )
    x_coordinate = FloatField()
    y_coordinate = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
