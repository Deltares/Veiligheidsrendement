from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestMeasureCombineController:
    def _create_sh_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
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

    def _create_sg_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
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

    def _create_section(self) -> SectionAsInput:
        return SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            measures=[],
            combined_measures=None,
        )

    def test_combine_combinable_partial_measures(self):
        @dataclass
        class MockMeasure:
            combine_type: CombinableTypeEnum | None

        # 1. Define input
        _measures = [
            MockMeasure(CombinableTypeEnum.COMBINABLE),
            MockMeasure(CombinableTypeEnum.PARTIAL),
        ]
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.PARTIAL]
        }

        # 2. Run test
        _combinations = CombineMeasuresController._combine_measures(
            _measures, _allowed_combinations
        )

        # 3. Verify expectations
        assert len(_combinations) == 2
        assert (
            len(
                [
                    c
                    for c in _combinations
                    if c.primary.combine_type == CombinableTypeEnum.COMBINABLE
                ]
            )
            == 2
        )
        assert len([c for c in _combinations if c.secondary is None]) == 1
        assert (
            len(
                [
                    c
                    for c in _combinations
                    if (
                        c.secondary is not None
                        and c.secondary.combine_type == CombinableTypeEnum.PARTIAL
                    )
                ]
            )
            == 1
        )

    def test_combine_combinable_revetment_measures(self):
        @dataclass
        class MockMeasure:
            combine_type: CombinableTypeEnum | None

        # 1. Define input
        _measures = [
            MockMeasure(CombinableTypeEnum.COMBINABLE),
            MockMeasure(CombinableTypeEnum.REVETMENT),
        ]
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.REVETMENT],
            CombinableTypeEnum.REVETMENT: [None],
        }

        # 2. Run test
        _combinations = CombineMeasuresController._combine_measures(
            _measures, _allowed_combinations
        )

        # 3. Verify expectations
        assert len(_combinations) == 3
        assert (
            len(
                [
                    c
                    for c in _combinations
                    if c.primary.combine_type == CombinableTypeEnum.COMBINABLE
                ]
            )
            == 2
        )
        assert (
            len(
                [
                    c
                    for c in _combinations
                    if c.primary.combine_type == CombinableTypeEnum.REVETMENT
                ]
            )
            == 1
        )
        assert len([c for c in _combinations if c.secondary is None]) == 2
        assert (
            len(
                [
                    c
                    for c in _combinations
                    if (
                        c.secondary is not None
                        and c.secondary.combine_type == CombinableTypeEnum.REVETMENT
                    )
                ]
            )
            == 1
        )

    def test_combining_sh_measures(self):
        # 1. Define input
        _section = self._create_section()
        _section.measures = [
            self._create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            self._create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            self._create_sh_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                CombinableTypeEnum.FULL,
            ),
            self._create_sh_measure(
                MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT
            ),
        ]

        # 2. Run test
        _combine_controller = CombineMeasuresController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == 7

    def test_combining_sg_measures(self):
        # 1. Define input
        _section = SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            measures=[],
            combined_measures=None,
        )
        _section.measures = [
            self._create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            self._create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
            ),
            self._create_sg_measure(
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                CombinableTypeEnum.FULL,
            ),
            self._create_sg_measure(
                MeasureTypeEnum.VERTICAL_GEOTEXTILE, CombinableTypeEnum.PARTIAL
            ),
        ]

        # 2. Run test
        _combine_controller = CombineMeasuresController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == 5
