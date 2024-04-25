import logging
from math import isnan

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.course_sand_barrier_measure_calculator import (
    CourseSandBarrierMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.fallback_measure_calculator import (
    FallbackMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.heavescreen_measure_calculator import (
    HeavescreenMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_geotextile_measure_calculator import (
    VerticalGeotextileMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_base import (
    VerticalPipingMeasureCalculatorBase,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class VerticalPipingMeasureCalculatorFactory:
    @staticmethod
    def get_calculator(
        dike_traject: DikeTrajectInfo,
        dike_section: DikeSection,
        measure: MeasureProtocol,
    ) -> VerticalPipingMeasureCalculatorProtocol:
        """
        Gets an instance of a `VerticalPipingMeasureCalculatorProtocol` depending on the provided arguments.
        At the moment the selection criteria depends mostly on the dike's section cover layer thicknes (`d_cover`).

        Args:
            dike_traject (DikeTrajectInfo): Dike traject in which the dike section takes place.
            dike_section (DikeSection): Dike section properties.
            measure (MeasureProtocol): Measure to be applied to the dike's section.

        Returns:
            VerticalPipingMeasureCalculatorProtocol: Calculator instance appropriate for the provided arguments.
        """

        def get_calculator_type() -> type[VerticalPipingMeasureCalculatorBase]:
            _d_cover = dike_section.cover_layer_thickness
            if _d_cover < 2:
                return CourseSandBarrierMeasureCalculator
            elif _d_cover >= 2 and _d_cover < 4:
                return VerticalGeotextileMeasureCalculator
            elif _d_cover >= 4 and _d_cover < 6:
                return HeavescreenMeasureCalculator
            elif _d_cover > 6:
                logging.warning(
                    "`d_cover` value is `%s`. When `d_cover > 6m` the probability of piping should be assumed minimal.",
                    _d_cover,
                )
                return FallbackMeasureCalculator

            logging.error(
                "`d_cover` value is `%s`. Using a calculator that assumes minimal probability of piping.",
                _d_cover,
            )
            return FallbackMeasureCalculator

        return get_calculator_type().from_measure_section_traject(
            measure=measure, dike_section=dike_section, traject_info=dike_traject
        )
