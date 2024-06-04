import math
from typing import Any

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)


class SgMeasureImporter(MeasureAsInputBaseImporter):
    @property
    def _measure_as_input_type(self) -> type[SgMeasure]:
        return SgMeasure

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
            _dberm = concrete_params_dict["dberm"]
            return math.isnan(_l_stab_screen) and (
                math.isclose(_dberm, 0) or math.isnan(_dberm)
            )

        if is_initial_cost_measure() and self._measure_result.measure_type in [
            MeasureTypeEnum.SOIL_REINFORCEMENT,
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
        ]:
            return section_cost
        return 0

    def _get_concrete_parameters_as_dictionary(
        self, section_cost: float, investment_year: int
    ) -> dict[str, Any]:
        _concrete_params_dict = {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in [
                "dberm",
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
