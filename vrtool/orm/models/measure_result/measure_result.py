from __future__ import annotations

from peewee import ForeignKeyField

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResult(OrmBaseModel):
    measure_per_section = ForeignKeyField(
        MeasurePerSection,
        backref="measure_per_section_result",
        on_delete="CASCADE",
    )

    class Meta:
        table_name = _get_table_name(__qualname__)

    @property
    def measure_type(self) -> MeasureTypeEnum:
        """
        Gets the related `MeasureType` table entry and maps it into the
        corresponding `vrtool.core` enum.

        Returns:
            MeasureTypeEnum: the mapped `MeasureTypeEnum`.
        """
        return MeasureTypeEnum.get_enum(
            self.measure_per_section.measure.measure_type.name
        )

    @property
    def combinable_type(self) -> CombinableTypeEnum:
        """
        Gets the related `CombinableType` table entry and maps it into the
        corresponding `vrtool.core` enum.

        Returns:
            CombinableTypeEnum: The mapped `CombinableTypeEnum`.
        """
        return CombinableTypeEnum.get_enum(
            self.measure_per_section.measure.combinable_type.name.upper()
        )

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
