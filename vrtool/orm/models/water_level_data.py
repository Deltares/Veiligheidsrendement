from peewee import ForeignKeyField, IntegerField, FloatField
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData

class WaterlevelData(OrmBaseModel):
    section_data = ForeignKeyField(SectionData, backref="water_level_data_list")
    water_level_location_id = IntegerField(null=False)
    year = IntegerField(null=False)
    water_level = FloatField(null=False)
    beta = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
