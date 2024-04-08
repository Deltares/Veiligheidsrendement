from __future__ import annotations

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

    def get_parameter_value(self, parameter_name: str) -> float:
        """
        Gets the value, or `float("nan")` when not found, from the list of parameters (`MeasureResultParameter`)
        which is accessed through backreference (`measure_result_parameters`).

        Args:
            parameter_name (str): Name of the parameter to look for.

        Returns:
            float: The value found or `float("nan")` when not.
        """
        return next(
            (
                _mrp.value
                for _mrp in self.measure_result_parameters
                if _mrp.name.lower() == parameter_name.lower()
            ),
            float("nan"),
        )
