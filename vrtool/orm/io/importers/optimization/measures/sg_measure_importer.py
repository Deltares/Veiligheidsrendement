import math
from collections import defaultdict
from typing import Any

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)


class SgMeasureImporter(MeasureAsInputBaseImporter):
    @property
    def _measure_as_input_type(self) -> type[SgMeasure]:
        return SgMeasure

    @staticmethod
    def set_initial_cost(measure_as_input_collection: list[MeasureAsInputProtocol]):
        def is_initial_cost_measure(sg_measure: SgMeasure) -> bool:
            if sg_measure.year != 0:
                return False
            return math.isclose(sg_measure.dberm, 0) or math.isnan(sg_measure.dberm)

        _cost_dictionary = defaultdict(lambda: defaultdict(lambda: 0.0))
        for _sg_measure in filter(is_initial_cost_measure, measure_as_input_collection):
            _cost_dictionary[_sg_measure.measure_type][
                _sg_measure.l_stab_screen
            ] = _sg_measure.cost

        for _sg_measure in measure_as_input_collection:
            _base_cost = _cost_dictionary[_sg_measure.measure_type][
                _sg_measure.l_stab_screen
            ]
            if _sg_measure.measure_type not in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
            ]:
                _base_cost = 0.0
            _sg_measure.base_cost = _base_cost

    def _get_concrete_parameters_as_dictionary(self) -> dict[str, Any]:
        _concrete_params_dict = {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in [
                "dberm",
                "l_stab_screen",
            ]
        }

        return _concrete_params_dict | dict(base_cost=0.0)
