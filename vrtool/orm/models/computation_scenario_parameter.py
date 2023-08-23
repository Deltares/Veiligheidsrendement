from peewee import CharField, FloatField, ForeignKeyField

from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class ComputationScenarioParameter(OrmBaseModel):
    computation_scenario = ForeignKeyField(
        ComputationScenario, backref="computation_scenario_parameters"
    )
    parameter = CharField(max_length=_max_char_length)
    value = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
