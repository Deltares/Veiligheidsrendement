from peewee import CharField, FloatField, ForeignKeyField

from vrtool.orm.models.optimization.optimization_type import OptimizationType
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class OptimizationRun(OrmBaseModel):
    name = CharField(max_length=_max_char_length, unique=True)
    discount_rate = FloatField()
    optimization_type = ForeignKeyField(OptimizationType, backref="optimization_runs")

    class Meta:
        table_name = _get_table_name(__qualname__)
