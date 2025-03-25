from dataclasses import dataclass, field
from typing import Callable, Iterable

import pytest

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


class TestCombineMeasuresController:
    def _create_sh_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> ShMeasure:
        # For now we don't really care which id they get.
        return ShMeasure(
            measure_result_id=42,
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=0,
            base_cost=0,
            year=0,
            discount_rate=0,
            mechanism_year_collection=MechanismPerYearProbabilityCollection(
                [
                    MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.1),
                    MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.2),
                    MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.5),
                    MechanismPerYear(MechanismEnum.REVETMENT, 20, 0.5),
                ]
            ),
            beta_target=0,
            transition_level=0,
            dcrest=0,
            l_stab_screen=float("nan"),
        )

    def _create_revetment_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> ShMeasure:
        # For now we don't really care which id they get.
        return ShMeasure(
            measure_result_id=42,
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=0,
            base_cost=0,
            year=0,
            discount_rate=0,
            mechanism_year_collection=MechanismPerYearProbabilityCollection(
                [
                    MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.5),
                    MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.5),
                    MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.1),
                    MechanismPerYear(MechanismEnum.REVETMENT, 20, 0.2),
                ]
            ),
            beta_target=0,
            transition_level=0,
            dcrest=0,
            l_stab_screen=float("nan"),
        )

    def _create_sg_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> SgMeasure:
        # For now we don't really care which id they get.
        return SgMeasure(
            measure_result_id=42,
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=0,
            base_cost=0,
            year=0,
            discount_rate=0,
            mechanism_year_collection=MechanismPerYearProbabilityCollection(
                [
                    MechanismPerYear(MechanismEnum.PIPING, 0, 0.1),
                    MechanismPerYear(MechanismEnum.PIPING, 20, 0.2),
                    MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.5),
                    MechanismPerYear(MechanismEnum.STABILITY_INNER, 20, 0.5),
                ]
            ),
            dberm=0,
            l_stab_screen=float("nan"),
        )

    def _create_section(self) -> SectionAsInput:
        return SectionAsInput(
            section_name="Section1",
            traject_name="Traject1",
            flood_damage=0,
            section_length=42,
            a_section_piping=2.4,
            a_section_stability_inner=4.2,
            measures=[],
        )

    @pytest.fixture(name="combinable_measures_factory")
    def _get_combinable_measures_factory(
        self,
    ) -> Iterable[
        Callable[
            [list[CombinableTypeEnum], list[MechanismPerYear]],
            list[MeasureAsInputProtocol],
        ]
    ]:
        @dataclass(kw_only=True)
        class ShMeasureGeotechnical(ShMeasure):
            beta_target: float = float("nan")
            transition_level: float = float("nan")
            dcrest: float = float("nan")

        def create_geotechnical_measure(
            measure_as_input_type: type[MeasureAsInputProtocol],
            combine_type: CombinableTypeEnum,
            mechanism_year_collection: list[MechanismPerYear],
        ) -> MeasureAsInputProtocol:
            return measure_as_input_type(
                measure_result_id=-1,
                measure_type=None,
                combine_type=combine_type,
                cost=float("nan"),
                base_cost=float("nan"),
                discount_rate=float("nan"),
                year=-1,
                l_stab_screen=float("nan"),
                mechanism_year_collection=MechanismPerYearProbabilityCollection(
                    mechanism_year_collection
                ),
            )

        def create_measures(
            combinable_types: list[CombinableTypeEnum],
            mechanism_year_collection: list[MechanismPerYear],
        ) -> list[MeasureAsInputProtocol]:
            return [
                create_geotechnical_measure(
                    ShMeasureGeotechnical,
                    _combinable_type,
                    mechanism_year_collection,
                )
                for _combinable_type in combinable_types
            ]

        yield create_measures

    def test_combine_combinable_partial_measures(
        self,
        combinable_measures_factory: Callable[
            [list[CombinableTypeEnum], list[MechanismPerYear]],
            list[MeasureAsInputProtocol],
        ],
    ):
        # 1. Define input
        _measures = combinable_measures_factory(
            [CombinableTypeEnum.COMBINABLE, CombinableTypeEnum.PARTIAL],
            [
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.5),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 20, 0.5),
                MechanismPerYear(MechanismEnum.PIPING, 0, 0.5),
                MechanismPerYear(MechanismEnum.PIPING, 20, 0.5),
            ],
        )
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.PARTIAL]
        }
        _initial_assessment = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.PIPING, 0, 0.5),
                MechanismPerYear(MechanismEnum.PIPING, 20, 0.5),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.5),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 20, 0.5),
            ]
        )
        _expected_combinations = len(_measures)

        # 2. Run test
        _combinations = CombineMeasuresController.combine_measures(
            _measures, _allowed_combinations, _initial_assessment
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
                        c.secondary
                        and c.secondary.combine_type == CombinableTypeEnum.PARTIAL
                    )
                ]
            )
            == 1
        )

    def test_combine_revetment_measures(
        self,
        combinable_measures_factory: Callable[
            [list[CombinableTypeEnum], list[MechanismPerYear]],
            list[MeasureAsInputProtocol],
        ],
    ):

        # 1. Define input
        _measures = combinable_measures_factory(
            [CombinableTypeEnum.COMBINABLE, CombinableTypeEnum.REVETMENT],
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.5),
                MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 20, 0.5),
            ],
        )
        _allowed_combinations = {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.REVETMENT],
            CombinableTypeEnum.REVETMENT: [None],
        }
        _initial_assessment = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.5),
                MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 20, 0.5),
            ]
        )
        _expected_combinations = len(_measures) + 1

        # 2. Run test
        _combinations = CombineMeasuresController.combine_measures(
            _measures, _allowed_combinations, _initial_assessment
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
                        c.secondary
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
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION, CombinableTypeEnum.PARTIAL
            ),
            self._create_revetment_measure(
                MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT
            ),
        ]
        _expected_combinations = (
            len(_measures) + 1
        )  # 2x combinable, 1x full, 2x combinable with revetment, 1x full with revetment
        _section.measures = _measures
        _section.initial_assessment = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.5),
                MechanismPerYear(MechanismEnum.OVERFLOW, 20, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.5),
                MechanismPerYear(MechanismEnum.REVETMENT, 20, 0.5),
            ]
        )

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
            section_length=42,
            a_section_piping=2.4,
            a_section_stability_inner=4.2,
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
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION, CombinableTypeEnum.PARTIAL
            ),
            self._create_sg_measure(
                MeasureTypeEnum.REVETMENT, CombinableTypeEnum.REVETMENT
            ),
        ]
        _expected_combinations = len(
            _measures
        )  # 2x combinable soil + 1x full soil + 2x combinable with partial
        _section.measures = _measures
        _section.initial_assessment = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.PIPING, 0, 0.5),
                MechanismPerYear(MechanismEnum.PIPING, 20, 0.5),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.5),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 20, 0.5),
            ]
        )

        # 2. Run test
        _combine_controller = CombineMeasuresController(_section)
        _combined_measures = _combine_controller.combine()

        # 3. Verify expectations
        assert len(_combined_measures) == _expected_combinations
