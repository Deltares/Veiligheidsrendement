from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class CustomMeasureDetail(OrmBaseModel):
    """
    A (logical) custom measure is defined by a set of records that share the same
    `measure_id`.
    We can derive the `section_id` from the `mechanism_per_section_id` foreign key.
    These are represented as a `CustomMeasureDetail`.
    """

    measure = ForeignKeyField(
        Measure, backref="custom_measure_details", on_delete="CASCADE"
    )
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="custom_measure_details", on_delete="CASCADE"
    )
    cost = FloatField(default=float("nan"), null=True)
    beta = FloatField(default=float("nan"), null=True)
    time = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)

    @property
    def measure_per_section(self) -> MeasurePerSection | None:
        """
        Gets the related `MeasurePerSection` orm object as the parent
        `Measure` object could be related to different `SectionData`.

        Returns:
            MeasurePerSection | None: The relation where this
            `CustomMeasureDetail` is applied.
        """
        return MeasurePerSection.get_or_none(
            (MeasurePerSection.measure == self.measure)
            & (MeasurePerSection.section == self.mechanism_per_section.section)
        )
