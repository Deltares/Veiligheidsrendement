from dataclasses import dataclass

import pytest as py

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta

_END_YEAR = 50
_mechm = MechanismEnum.OVERFLOW


@dataclass
class MockShMeasure(ShMeasure):
    measure_type: MeasureTypeEnum
    measure_result_id: int = 42
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
    measure_result_id: int = 42
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
            flood_damage=0,
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
                MockShMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN),
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

    def _get_revetment_measure(
        self, year: int, revetment_params: list[float], betas: list[float]
    ) -> ShMeasure:
        """
        get a single (revetment) measure

        Args:
            year (int): investment year for this measure
            revetment_params (list[float]): list with: beta_target, transition_level, dcrest
            betas (list[float]): list with beta values for years 0 and 50

        Returns:
            ShMeasure: a revetment measure based on the given arguments
        """
        _mech_per_year1 = MechanismPerYear(_mechm, 0, beta_to_pf(betas[0]))
        _mech_per_year2 = MechanismPerYear(_mechm, _END_YEAR, beta_to_pf(betas[1]))
        _collection = MechanismPerYearProbabilityCollection(
            [_mech_per_year1, _mech_per_year2]
        )
        _dummy_cost = 999.0
        _dummy_discount_rate = 0.05
        _measure = ShMeasure(
            42,
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            _dummy_cost,
            _dummy_discount_rate,
            year,
            _collection,
            revetment_params[0],
            revetment_params[1],
            revetment_params[2],
        )
        return _measure

    def _get_initial_probabilities(
        self, beta_yr0: float, beta_end_year: float
    ) -> MechanismPerYearProbabilityCollection:
        _mech1_year1_prob = MechanismPerYear(
            MechanismEnum.OVERFLOW, 0, beta_to_pf(beta_yr0)
        )
        _mech1_year2_prob = MechanismPerYear(
            MechanismEnum.OVERFLOW, _END_YEAR, beta_to_pf(beta_end_year)
        )

        _mech2_year1_prob = MechanismPerYear(
            MechanismEnum.PIPING, 0, beta_to_pf(beta_yr0)
        )
        _mech2_year2_prob = MechanismPerYear(
            MechanismEnum.PIPING, _END_YEAR, beta_to_pf(beta_end_year)
        )
        _mechanism_year_collection = MechanismPerYearProbabilityCollection(
            [_mech1_year1_prob, _mech1_year2_prob, _mech2_year1_prob, _mech2_year2_prob]
        )

        return _mechanism_year_collection

    def test_investment_year_basic(self):
        """
        Minimal test: two zero measures and an actual measure
        Reference values can be obtained using linear interpolation
        """
        # setup
        _yr1 = 20
        _prob_diff = 0.5
        _prob_zero = [4.0, 4.0 - _prob_diff]
        _prob_measure = [5.0, 5.0 - _prob_diff]
        _measure = self._get_revetment_measure(_yr1, [4.0, 2.0, 0.0], _prob_measure)
        _initial = self._get_initial_probabilities(_prob_zero[0], _prob_zero[1])
        _measures = [_measure]
        _section_as_input = SectionAsInput(
            section_name="section1",
            traject_name="traject1",
            flood_damage=0.0,
            measures=_measures,
            initial_assessment=_initial,
        )

        # run test
        _section_as_input.update_measurelist_with_investment_year()

        # check results

        # year 0 for the first measure is copied from the zero measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, 0)
        assert pf_to_beta(_pf) == py.approx(_prob_zero[0])

        # year 19 for the first measure is an interpolated value copied from the zero measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(
            MechanismEnum.OVERFLOW, _yr1 - 1
        )
        _beta_expected = _prob_zero[0] - _prob_diff * (_yr1 - 1) / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # year 20 for the first measure is an interpolated value from this measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, _yr1)
        _beta_expected = _prob_measure[0] - _prob_diff * _yr1 / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # year 50 for the first measure is a copy of the last year
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, _END_YEAR)
        _beta_expected = _prob_measure[1]
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # all measures are extended with two years:
        _ref = {0, _yr1 - 1, _yr1, _END_YEAR}
        for m in _measures:
            _yrs = m.mechanism_year_collection.get_years(_mechm)
            assert _yrs == _ref

    def test_two_investment_years(self):
        """
        Test two zero measures and two actual measures that only differ in investment year
        Reference values can be obtained using linear interpolation
        """
        # setup
        _yr1 = 20
        _yr2 = 30
        _prob_diff = 0.5
        _prob_zero = [4.0, 4.0 - _prob_diff]
        _prob_measure_a = [5.0, 5.0 - _prob_diff]
        _prob_measure_b = [6.0, 6.0 - _prob_diff]
        _initial_probabilities = self._get_initial_probabilities(
            _prob_zero[0], _prob_zero[1]
        )
        _measures = [
            self._get_revetment_measure(_yr1, [4.0, 2.0, 0.0], _prob_measure_a),
            self._get_revetment_measure(_yr2, [5.0, 2.0, 0.0], _prob_measure_b),
        ]
        _section_as_input = SectionAsInput(
            section_name="section1",
            traject_name="traject1",
            flood_damage=0.0,
            measures=_measures,
            initial_assessment=_initial_probabilities,
        )

        # run test
        _section_as_input.update_measurelist_with_investment_year()

        # check results

        # year 0 for the first measure is copied from the zero measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, 0)
        assert pf_to_beta(_pf) == py.approx(_prob_zero[0])

        # year 19 for the first measure is an interpolated value copied from the zero measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, _yr1 - 1)
        _beta_expected = _prob_zero[0] - _prob_diff * (_yr1 - 1) / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # year 20 for the first measure is an interpolated value from this measure:
        _pf = _measures[0].mechanism_year_collection.get_probability(_mechm, _yr1)
        _beta_expected = _prob_measure_a[0] - _prob_diff * _yr1 / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # year 29 for the second measure is an interpolated value copied from the zero measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(_mechm, _yr2 - 1)
        _beta_expected = _prob_zero[0] - _prob_diff * (_yr2 - 1) / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # year 30 for the second measure is an interpolated value from this measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(_mechm, _yr2)
        _beta_expected = _prob_measure_b[0] - _prob_diff * _yr2 / _END_YEAR
        assert pf_to_beta(_pf) == py.approx(_beta_expected)

        # all measures are extended with four years:
        _ref = {0, _yr1 - 1, _yr1, _yr2 - 1, _yr2, _END_YEAR}
        for m in _measures:
            _yrs = m.mechanism_year_collection.get_years(_mechm)
            assert _yrs == _ref

    def test_get_combination_for_aggregate(self):
        # 1. Define test data
        _section = self._get_section_with_combinations()
        _aggregated_measure_combination = AggregatedMeasureCombination(
            sh_combination=_section.sh_combinations[0],
            sg_combination=_section.sg_combinations[0],
            year=0,
        )

        # 2. Run test
        _sg_idx, _sh_idx = _section.get_combination_for_aggregate(
            _aggregated_measure_combination
        )

        # 3. Verify expectations
        assert _sg_idx == 2
        assert _sh_idx == 0
