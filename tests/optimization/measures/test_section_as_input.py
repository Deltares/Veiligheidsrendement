from dataclasses import dataclass

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class MockShMeasure(ShMeasure):
    measure_type: MeasureTypeEnum
    combine_type: None = None
    cost: float = 0
    discount_rate: float = 0
    year: int = 0
    lcc: float = 0
    mechanism_year_collection: None = None
    beta_target: float = 0
    transition_level: float = 0
    dcrest: float = 0


@dataclass
class MockSgMeasure(SgMeasure):
    measure_type: MeasureTypeEnum
    combine_type: None = None
    cost: float = 0
    discount_rate: float = 0
    year: int = 0
    lcc: float = 0
    mechanism_year_collection: None = None
    dberm: float = 0
    dcrest: float = 0


class TestSectionAsInput:

    def _get_section_with_measures(self) -> SectionAsInput:
        return SectionAsInput(
            section_name="section_name",
            traject_name="traject_name",
            measures=[
                MockShMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT),
                MockShMeasure(MeasureTypeEnum.REVETMENT),
                MockSgMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN),
                MockSgMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE),
            ],
        )

    def _get_section_with_combinations(self) -> SectionAsInput:
        _section = self._get_section_with_measures()
        _section.combined_measures = [
            CombinedMeasure.from_input(
                MockShMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT),
                None,
            ),
            CombinedMeasure.from_input(
                MockSgMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN),
                None,
            ),
        ]
        return _section

    def test_get_sh_measures(self):
        # 1. Define test data
        _section = self._get_section_with_measures()

        # 2. Run test
        _sh_measures = _section.sh_measures

        # 3. Verify expectations
        assert len(_sh_measures) == 2
        assert any(
            x.measure_type == MeasureTypeEnum.SOIL_REINFORCEMENT for x in _sh_measures
        )
        assert any(x.measure_type == MeasureTypeEnum.REVETMENT for x in _sh_measures)

    def test_get_sg_measures(self):
        # 1. Define test data
        _section = self._get_section_with_measures()

        # 2. Run test
        _sg_measures = _section.sg_measures

        # 3. Verify expectations
        assert len(_sg_measures) == 2
        assert any(
            x.measure_type == MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN
            for x in _sg_measures
        )
        assert any(
            x.measure_type == MeasureTypeEnum.VERTICAL_GEOTEXTILE for x in _sg_measures
        )

    def test_get_sh_combinations(self):
        # 1. Define test data
        _section = self._get_section_with_combinations()

        # 2. Run test
        _sh_combinations = _section.sh_combinations

        # 3. Verify expectations
        assert len(_sh_combinations) == 1
        assert (
            _sh_combinations[0].primary.measure_type
            == MeasureTypeEnum.SOIL_REINFORCEMENT
        )

    def test_get_sg_combinations(self):
        # 1. Define test data
        _section = self._get_section_with_combinations()

        # 2. Run test
        _sg_combinations = _section.sg_combinations

        # 3. Verify expectations
        assert len(_sg_combinations) == 1
        assert (
            _sg_combinations[0].primary.measure_type
            == MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN
        )
