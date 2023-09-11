from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResult(OrmBaseModel):
    beta = FloatField()
    time = IntegerField()
    cost = FloatField()
    measure_per_section = ForeignKeyField(
        MeasurePerSection, backref="measure_per_section_result"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
