from peewee import ForeignKeyField, FloatField, CharField

from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name, _max_char_length


class MeasureResultParameter(OrmBaseModel):
    name = CharField(unique=True, max_length=_max_char_length)
    value = FloatField()
    measure_result_id = ForeignKeyField(MeasureResult, backref="measure_result_parameters")

    class Meta:
        table_name = _get_table_name(__qualname__)
