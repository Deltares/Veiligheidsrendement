from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MechanismTable(OrmBaseModel):
    computation_scenario = ForeignKeyField(
        ComputationScenario, backref="mechanism_tables"
    )
    year = IntegerField(null=False)
    value = FloatField(null=False)
    beta = FloatField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
