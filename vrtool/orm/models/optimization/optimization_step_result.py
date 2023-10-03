from peewee import ForeignKeyField, IntegerField, FloatField
from vrtool.orm.models.mechanism_per_section import MechanismPerSection

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class OptimizationStepResult(OrmBaseModel):
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="mechanism_optimization_step_results"
    )
    beta = FloatField()
    time = IntegerField()
    lcc = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
