from peewee import ForeignKeyField

from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResult(OrmBaseModel):
    measure_per_section = ForeignKeyField(
        MeasurePerSection, backref="measure_per_section_result"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)

    @property
    def section_result(self):
        """
        Gets the `MeasureResultSection` (one-to-one relationship)
         for this `MeasureResult.
        """
        return self.sections_measure_result.get()
