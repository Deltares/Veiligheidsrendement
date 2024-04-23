from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.vertical_piping_measures.vertical_piping_measure_calculator_factory import (
    VerticalPipingMeasureCalculatorFactory,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class VerticalPipingSolutionMeasure(MeasureProtocol):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        # No influence on overflow and stability
        # Only 1 parameterized version with a lifetime of 50 years
        self.measures = {}
        self.measures["VZG"] = "yes"
        # self.measures["Cost"] = (
        #     self.unit_costs["Vertical Geotextile"] * dike_section.Length
        # )
        # self.measures["Reliability"] = self._get_configured_section_reliability(
        #     dike_section, traject_info
        # )
        _calculator = VerticalPipingMeasureCalculatorFactory.get_calculator(
            traject_info, dike_section, self
        )
        self.measures["Cost"] = _calculator.calculate_cost(self.unit_costs)
        self.measures["Reliability"] = _calculator.calculate_reliability()
