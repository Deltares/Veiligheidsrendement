from vrtool.orm.models.base_model import BaseModel, _get_table_name, _max_char_length
from peewee import ForeignKeyField, CharField, IntegerField
from vrtool.orm.models.section_data import SectionData

class Buildings(BaseModel):
    section_data = ForeignKeyField(SectionData, backref="buildings_list")

    distance_from_toe = CharField(null=False, max_length=_max_char_length)
    number_of_buildings = IntegerField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)