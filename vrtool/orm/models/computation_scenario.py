from peewee import CharField, FloatField, ForeignKeyField

from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class ComputationScenario(OrmBaseModel):
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="computation_scenarios", on_delete="CASCADE"
    )
    computation_type = ForeignKeyField(
        ComputationType, backref="computation_scenarios", on_delete="CASCADE"
    )
    computation_name = CharField(max_length=_max_char_length)
    scenario_name = CharField(max_length=_max_char_length)
    scenario_probability = FloatField(default=float("nan"), null=True)
    probability_of_failure = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
