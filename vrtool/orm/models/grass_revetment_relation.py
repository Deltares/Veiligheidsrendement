from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class GrassRevetmentRelation(OrmBaseModel):
    computation_scenario = ForeignKeyField(
        ComputationScenario, backref="grass_revetment_relations"
    )
    year = IntegerField()
    transition_level = FloatField()
    beta = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
