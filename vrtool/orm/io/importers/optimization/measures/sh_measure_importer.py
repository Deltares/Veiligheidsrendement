import math
from collections import defaultdict
from typing import Any

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)


class ShMeasureImporter(MeasureAsInputBaseImporter):
    @property
    def _measure_as_input_type(self) -> type[ShMeasure]:
        return ShMeasure

    @staticmethod
    def set_initial_cost(measure_as_input_collection: list[MeasureAsInputProtocol]):
        def is_initial_cost_measure(sh_measure: ShMeasure) -> bool:
            if sh_measure.year != 0:
                return False
            return math.isnan(sh_measure.l_stab_screen) and (
                math.isclose(sh_measure.dcrest, 0) or math.isnan(sh_measure.dcrest)
            )

        _cost_dictionary = defaultdict(lambda: 0.0)
        for _sh_measure in filter(is_initial_cost_measure, measure_as_input_collection):
            _cost_dictionary[_sh_measure.measure_type] = _sh_measure.cost

        for _sh_measure in measure_as_input_collection:
            _base_cost = _cost_dictionary[_sh_measure.measure_type]
            if _sh_measure.measure_type not in [
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
                MeasureTypeEnum.DIAPHRAGM_WALL,
                MeasureTypeEnum.STABILITY_SCREEN,
            ]:
                _base_cost = 0.0
            _sh_measure.base_cost = _base_cost

    def _get_concrete_parameters_as_dictionary(self) -> dict[str, Any]:
        _concrete_params_dict = {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in [
                "beta_target",
                "transition_level",
                "dcrest",
                "l_stab_screen",
            ]
        }
        return _concrete_params_dict | dict(base_cost=0.0)
