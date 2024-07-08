from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class WaterlevelData(OrmBaseModel):
    section_data = ForeignKeyField(
        SectionData, backref="water_level_data_list", on_delete="CASCADE"
    )
    water_level_location_id = IntegerField(null=True)
    year = IntegerField()
    water_level = FloatField()
    beta = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
