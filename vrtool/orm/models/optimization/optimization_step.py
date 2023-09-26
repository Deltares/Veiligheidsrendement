from peewee import IntegerField, ForeignKeyField
from vrtool.orm.models.optimization.optimization_run import OptimizationRun

from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
)


class OptimizationStep(OrmBaseModel):
    optimization_run = ForeignKeyField(OptimizationRun, backref="optimization_steps")
    step_number = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
