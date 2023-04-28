from vrtool.orm.models.base_model import BaseModel, _get_table_name
from peewee import IntegerField, ForeignKeyField, FloatField
from vrtool.orm.models.computation_scenario import ComputationScenario

class MechanismTable(BaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="mechanism_tables")
    year = IntegerField(null=False)
    value = FloatField(null=False)
    beta = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
