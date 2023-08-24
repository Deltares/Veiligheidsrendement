from peewee import ForeignKeyField, FloatField

from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResult(OrmBaseModel):
    beta = FloatField()
    time = FloatField()
    cost = FloatField()
    measure_per_section = ForeignKeyField(MeasurePerSection, backref="measure_per_section_results")

    class Meta:
        table_name = _get_table_name(__qualname__)
