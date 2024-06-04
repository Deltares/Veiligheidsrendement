from typing import Any

from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)


class ShSgMeasureImporter(MeasureAsInputBaseImporter):
    @property
    def _measure_as_input_type(self) -> type[ShSgMeasure]:
        return ShSgMeasure

    def _get_concrete_parameters_as_dictionary(
        self, section_cost: float, investment_year: int
    ) -> dict[str, Any]:
        return {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in [
                "dberm",
                "dcrest",
                "l_stab_screen",
            ]
        }
