from peewee import ForeignKeyField

from vrtool.orm.models.custom_measure import CustomMeasure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class CustomMeasurePerMeasurePerSection(OrmBaseModel):
    measure_per_section = ForeignKeyField(
        MeasurePerSection,
        backref="custom_measures_per_measure_per_section",
        on_delete="CASCADE",
    )
    custom_measure = ForeignKeyField(
        CustomMeasure,
        backref="measure_per_sections_custom_measures",
        on_delete="CASCADE",
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
