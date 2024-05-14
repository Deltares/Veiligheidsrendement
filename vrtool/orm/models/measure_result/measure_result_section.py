from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResultSection(OrmBaseModel):
    measure_result = ForeignKeyField(
        MeasureResult, backref="measure_result_section", on_delete="CASCADE"
    )

    beta = FloatField()
    time = IntegerField()
    cost = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
