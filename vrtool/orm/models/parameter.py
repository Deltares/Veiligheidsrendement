from vrtool.orm.models.base_model import OrmBaseModel, _get_table_name, _max_char_length
from vrtool.orm.models.computation_scenario import ComputationScenario
from peewee import ForeignKeyField, CharField, FloatField

class Parameter(OrmBaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="parameters")
    parameter = CharField(unique=True, max_length=_max_char_length)
    value = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
