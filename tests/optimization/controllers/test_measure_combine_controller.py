from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.measure_combine_controller import (
    MeasureCombineController,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestMeasureCombineController:
    def test_combining_sh_measures(self):
        def _create_sh_measure(
            measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
        ) -> ShMeasure:
            return ShMeasure(
                measure_type=measure_type,
                combine_type=combinable_type,
                cost=0,
                year=0,
                lcc=0,
                mechanism_year_collection=None,
                beta_target=0,
                transition_level=0,
                dcrest=0,
            )

        # 1. Define input
        _section = SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            measures=[],
            combined_measures=None,
        )
        _section.measures = [
            _create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            _create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            _create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                CombinableTypeEnum.FULL,
            ),
            _create_sh_measure(MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT),
        ]

        # 2. Run test
        _combine_controller = MeasureCombineController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == 7

    def test_combining_sg_measures(self):
        def _create_sg_measure(
            measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
        ) -> SgMeasure:
            return SgMeasure(
                measure_type=measure_type,
                combine_type=combinable_type,
                cost=0,
                year=0,
                lcc=0,
                mechanism_year_collection=None,
                dcrest=0,
                dberm=0,
            )

        # 1. Define input
        _section = SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            measures=[],
            combined_measures=None,
        )
        _section.measures = [
            _create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            _create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            _create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                CombinableTypeEnum.FULL,
            ),
            _create_sg_measure(
                MeasureTypeEnum.VERTICAL_GEOTEXTILE, CombinableTypeEnum.PARTIAL
            ),
        ]

        # 2. Run test
        _combine_controller = MeasureCombineController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == 5
