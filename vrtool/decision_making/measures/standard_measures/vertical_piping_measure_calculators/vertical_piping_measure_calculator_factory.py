import logging

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
from vrtool.flood_defence_system.dike_section import DikeSection


class VerticalPipingMeasureCalculatorFactory:
    @staticmethod
    def get_calculator(
        dike_traject: DikeTrajectInfo,
        dike_section: DikeSection,
        measure: MeasureProtocol,
    ):
        _d_cover = dike_section.cover_layer_thickness
        if _d_cover < 2:
            return CourseSandBarrierMeasureCalculator.from_measure_section_traject(
                measure, dike_section, dike_traject
            )
        elif _d_cover >= 2 and _d_cover < 4:
            return VerticalGeotextileMeasureCalculator.from_measure_section_traject(
                measure, dike_section, dike_traject
            )
        elif _d_cover >= 4 and _d_cover < 6:
            return HeavescreenMeasureCalculator.from_measure_section_traject(
                measure, dike_section, dike_traject
            )
        elif _d_cover > 6:
            # TODO: When `d_cover > 6m` the probability of piping should be assumed minimal
            logging.warning(
                "`d_cover` value is `{}`. When `d_cover > 6m` the probability of piping should be assumed minimal.".format(
                    _d_cover
                )
            )
            return FallbackMeasureCalculator.from_measure_section_traject(
                measure, dike_section, dike_traject
            )
