from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class Buildings(OrmBaseModel):
    section_data = ForeignKeyField(SectionData, backref="buildings_list")

    distance_from_toe = FloatField()
    number_of_buildings = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
