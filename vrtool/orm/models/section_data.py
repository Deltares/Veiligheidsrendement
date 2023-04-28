from peewee import BooleanField, CharField, FloatField, ForeignKeyField

from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class SectionData(OrmBaseModel):
    dike_traject = ForeignKeyField(DikeTrajectInfo, backref="dike_sections")
    section_name = CharField(unique=True, max_length=_max_char_length)
    dijkpaal_start = CharField(null=True, max_length=_max_char_length)
    dijkpaal_end = CharField(null=True, max_length=_max_char_length)
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
