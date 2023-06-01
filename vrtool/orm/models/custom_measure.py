from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure import Measure
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class CustomMeasure(OrmBaseModel):
    # TODO: Rename to custom_measure as it's a 0.1 cardinality.
    measure = ForeignKeyField(Measure, backref="custom_measures", unique=True)
    mechanism = ForeignKeyField(Mechanism, backref="measures")
    cost = FloatField(default=float("nan"), null=True)
    beta = FloatField(default=float("nan"), null=True)
    year = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
