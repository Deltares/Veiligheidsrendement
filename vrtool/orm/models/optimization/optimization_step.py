from peewee import IntegerField, ForeignKeyField
from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)

from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
)


class OptimizationStep(OrmBaseModel):
    """
    This table represents the selection of a `ResultMeasure` as an optimization step result.
    The `step_number` can be repeated, as the `optimization_selected_measure` specifies the
    `OptimizationRun` of this step.
    """

    optimization_selected_measure = ForeignKeyField(
        OptimizationSelectedMeasure, backref="optimization_steps"
    )
    step_number = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
