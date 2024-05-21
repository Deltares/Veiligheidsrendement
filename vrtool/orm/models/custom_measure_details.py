from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure import Measure
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class CustomMeasureDetails(OrmBaseModel):
    """
    A (logical) custom measure is defined by a set of records that share the same measure_id.
    """

    measure = ForeignKeyField(
        Measure, backref="custom_measure_details", on_delete="CASCADE"
    )
    mechanism = ForeignKeyField(
        Mechanism, backref="custom_measure_details", on_delete="CASCADE"
    )
    cost = FloatField(default=float("nan"), null=True)
    beta = FloatField(default=float("nan"), null=True)
    year = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
