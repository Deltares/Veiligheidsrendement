from peewee import CharField, FloatField, ForeignKeyField

from vrtool.orm.models.custom_measure import CustomMeasure
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class CustomMeasureParameter(OrmBaseModel):
    custom_measure = ForeignKeyField(
        CustomMeasure, backref="custom_parameters", on_delete="CASCADE"
    )
    parameter = CharField(max_length=_max_char_length)
    value = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
