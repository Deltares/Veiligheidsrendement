import math
from typing import Any

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)


class ShMeasureImporter(MeasureAsInputBaseImporter):
    @property
    def _measure_as_input_type(self) -> type[ShMeasure]:
        return ShMeasure

    def _determine_start_cost(
        self,
        concrete_params_dict: dict[str, float],
        section_cost: float,
        investment_year: int,
    ) -> float:
        def is_initial_cost_measure() -> bool:
            if investment_year != 0:
                return False
            _l_stab_screen = concrete_params_dict["l_stab_screen"]
            _dcrest = concrete_params_dict["dcrest"]
            return math.isnan(_l_stab_screen) and (
                math.isclose(_dcrest, 0) or math.isnan(_dcrest)
            )

        if not is_initial_cost_measure() or self._measure_result.measure_type not in [
            MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]:
            return 0
        return section_cost

    def _get_concrete_parameters_as_dictionary(
        self, section_cost: float, investment_year: int
    ) -> dict[str, Any]:
        _concrete_params_dict = {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in [
                "beta_target",
                "transition_level",
                "dcrest",
                "l_stab_screen",
            ]
        }
        return (
            dict(
                start_cost=self._determine_start_cost(
                    _concrete_params_dict, section_cost, investment_year
                )
            )
            | _concrete_params_dict
        )
