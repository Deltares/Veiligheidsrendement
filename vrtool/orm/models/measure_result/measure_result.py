from __future__ import annotations
from peewee import ForeignKeyField, fn

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

    def get_parameter(self, parameter_name: str) -> float:
        _values = self.measure_result_parameters.where(
            fn.Lower(OrmMeasureResultParameter.name) == parameter_name.lower()
        ).select()
        return _values[0].value if any(_values) else float("nan")
