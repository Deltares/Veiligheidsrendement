from peewee import ForeignKeyField

from vrtool.orm.models.custom_measure_detail import CustomMeasureDetail
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class CustomMeasureDetailPerSection(OrmBaseModel):
    """
    This table represents the cross-reference table between
    `CustomMeasureDetail` and `MeasurePerSection`.
    For readability purposes it has been chosen to name it
    as `CustomMeasureDetailPerSection` rather than
    `CustomMeasureDetailPerMeasurePerSection`.
    """

    measure_per_section = ForeignKeyField(
        MeasurePerSection,
        backref="custom_measure_details_per_section",
        on_delete="CASCADE",
    )
    custom_measure_detail = ForeignKeyField(
        CustomMeasureDetail,
        backref="sections_per_custom_measure_detail",
        on_delete="CASCADE",
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
