from dataclasses import dataclass, field

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class MockMechanismYearProColl(MechanismPerYearProbabilityCollection):
    probabilities: list[MechanismPerYear] = field(default_factory=list)


@dataclass
class MockMeasure(MeasureAsInputProtocol):
    combine_type: CombinableTypeEnum
    mechanism_year_collection: MockMechanismYearProColl = MockMechanismYearProColl(
        [
            MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.9),
            MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.8),
        ]
    )


class TestCombineMeasuresController:
    def _create_sh_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> ShMeasure:
        return ShMeasure(
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=0,
            year=0,
            discount_rate=0,
            mechanism_year_collection=MockMechanismYearProColl(
                [
                    MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.9),
                    MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.8),
                ]
            ),
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
            discount_rate=0,
            mechanism_year_collection=MockMechanismYearProColl(
                [
                    MechanismPerYear(MechanismEnum.PIPING, 0, 0.7),
                    MechanismPerYear(MechanismEnum.PIPING, 20, 0.6),
                ]
            ),
            dcrest=0,
            dberm=0,
        )

    def _create_section(self) -> SectionAsInput:
        return SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            flood_damage=0,
            measures=[],
        )

    def test_combine_combinable_partial_measures(self):

        # 1. Define input
        _measures = [
            MockMeasure(CombinableTypeEnum.COMBINABLE),
            MockMeasure(CombinableTypeEnum.PARTIAL),
        ]
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.PARTIAL]
        }
        _expected_combinations = len(_measures)

        # 2. Run test
        _combinations = CombineMeasuresController.combine_measures(
            _measures, _allowed_combinations
        )

        # 3. Verify expectations
        assert len(_combinations) == _expected_combinations
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

        # 1. Define input
        _measures = [
            MockMeasure(CombinableTypeEnum.COMBINABLE),
            MockMeasure(CombinableTypeEnum.REVETMENT),
        ]
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.REVETMENT],
            CombinableTypeEnum.REVETMENT: [None],
        }
        _expected_combinations = len(_measures) + 1

        # 2. Run test
        _combinations = CombineMeasuresController.combine_measures(
            _measures, _allowed_combinations
        )

        # 3. Verify expectations
        assert len(_combinations) == _expected_combinations
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
        _measures = [
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
                MeasureTypeEnum.VERTICAL_GEOTEXTILE, CombinableTypeEnum.PARTIAL
            ),
            self._create_sh_measure(
                MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT
            ),
        ]
        _expected_combinations = len(_measures) + 1
        _section.measures = _measures

        # 2. Run test
        _combine_controller = CombineMeasuresController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == _expected_combinations

    def test_combining_sg_measures(self):
        # 1. Define input
        _section = SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            flood_damage=0,
            measures=[],
        )
        _measures = [
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
            self._create_sg_measure(
                MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT
            ),
        ]
        _expected_combinations = len(_measures)
        _section.measures = _measures

        # 2. Run test
        _combine_controller = CombineMeasuresController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == _expected_combinations
