from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class OptimizationStepResultMechanism(OrmBaseModel):
    optimization_step = ForeignKeyField(
        OptimizationStep,
        backref="optimization_step_results_mechanism",
        on_delete="CASCADE",
    )
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection,
        backref="mechanism_optimization_step_results",
        on_delete="CASCADE",
    )
    beta = FloatField()
    time = IntegerField()
    lcc = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
