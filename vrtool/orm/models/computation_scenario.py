from vrtool.orm.models.base_model import BaseModel, _get_table_name, _max_char_length
from peewee import ForeignKeyField, CharField, FloatField
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.computation_type import ComputationType

class ComputationScenario(BaseModel):
    mechanism_per_section = ForeignKeyField(MechanismPerSection, backref="computation_scenarios")
    computation_type = ForeignKeyField(ComputationType, backref="computation_scenarios")
    computation_name = CharField(null=False, max_length=_max_char_length)
    scenario_name = CharField(max_length=_max_char_length)
    scenario_probability = FloatField(null=False)
    probability_of_failure = FloatField()
    class Meta:
        table_name = _get_table_name(__qualname__)