from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.mechanism import Mechanism
from peewee import ForeignKeyField, FloatField, IntegerField

class CustomMeasure(OrmBaseModel):
    measure = ForeignKeyField(Measure, backref="custom_measures", unique=True)
    mechanism = ForeignKeyField(Mechanism, backref="measures")
    cost = FloatField(null=False)
    beta = FloatField(null=False)
    year = IntegerField(null=False)

    class Meta:
        table_name = _get_table_name(__qualname__)
