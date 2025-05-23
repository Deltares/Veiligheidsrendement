from peewee import ForeignKeyField, IntegerField

from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class OptimizationSelectedMeasure(OrmBaseModel):
    """
    Cross-reference table to represent the many-to-many relationship between
    `OptimizationRun` and `MeasureResult`.
    """

    optimization_run = ForeignKeyField(
        OptimizationRun, backref="optimization_run_measure_results", on_delete="CASCADE"
    )
    measure_result = ForeignKeyField(
        MeasureResult, backref="measure_result_optimization_runs", on_delete="CASCADE"
    )
    investment_year = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
