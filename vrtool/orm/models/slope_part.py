from peewee import FloatField, ForeignKeyField

from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class SlopePart(OrmBaseModel):
    computation_scenario = ForeignKeyField(ComputationScenario, backref="slope_parts")
    begin_part = FloatField()
    end_part = FloatField()
    top_layer_type = FloatField()
    top_layer_thickness = FloatField(null=True)
    tan_alpha = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
