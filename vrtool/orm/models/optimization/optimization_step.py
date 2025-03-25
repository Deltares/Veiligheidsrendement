from peewee import CharField, FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class OptimizationStep(OrmBaseModel):
    """
    This table represents the selection of a `ResultMeasure` as an optimization step result.
    The `step_number` can be repeated, as the `optimization_selected_measure` specifies the
    `OptimizationRun` of this step.
    """

    optimization_selected_measure = ForeignKeyField(
        OptimizationSelectedMeasure, backref="optimization_steps", on_delete="CASCADE"
    )
    step_number = IntegerField()
    step_type = CharField(max_length=_max_char_length)
    total_lcc = FloatField(null=True)
    total_risk = FloatField(null=True)

    class Meta:
        table_name = _get_table_name(__qualname__)
