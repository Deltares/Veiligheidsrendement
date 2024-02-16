import pytest as py

from vrtool.optimization.measures.set_investment_year import SetInvestmentYear
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class TestSetInvestmentYear:
    _end_year = 50

    def _get_measure(
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
        _mech_per_year1 = MechanismPerYear(
            MeasureTypeEnum.REVETMENT, 0, beta_to_pf(betas[0])
        )
        _mech_per_year2 = MechanismPerYear(
            MeasureTypeEnum.REVETMENT, self._end_year, beta_to_pf(betas[1])
        )
        _collection = MechanismPerYearProbabilityCollection(
            [_mech_per_year1, _mech_per_year2]
        )
        _dummy_cost = 999.0
        _dummy_lcc = 999.9
        _measure = ShMeasure(
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            _dummy_cost,
            year,
            _dummy_lcc,
            _collection,
            revetment_params[0],
            revetment_params[1],
            revetment_params[2],
        )
        return _measure

    def test_two_measures(self):
        # setup
        _yr1 = 20
        _prob_diff = 0.5
        _prob_zero = [4.0, 4.0 - _prob_diff]
        _measures = [
            self._get_measure( 0  , [0.0, 0.0, 0.0], _prob_zero),
            self._get_measure(_yr1, [4.0, 2.0, 0.0], [5.0, 4.5]),
        ]

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measures)

        # check results

        # year 0 for the second measure is copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 0)
        assert (pf_to_beta(_pf) == py.approx(_prob_zero[0]))

        # year 20 for the second measure is an interpolated value copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT,_yr1)
        _beta_expected = _prob_zero[0] - _prob_diff * _yr1 / self._end_year
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # all measures are extended with two years:
        _ref = {0, _yr1, _yr1 + 1, self._end_year}
        _yrs = _measures[0].mechanism_year_collection.get_years(MeasureTypeEnum.REVETMENT)
        assert _yrs == _ref
        _yrs = _measures[1].mechanism_year_collection.get_years(MeasureTypeEnum.REVETMENT)
        assert _yrs == _ref

    def test_three_measures(self):
        # setup
        _yr1 = 20
        _yr2 = 30
        _prob_diff = 0.5
        _prob_zero = [4.0, 4.0 - _prob_diff]
        _measures = [
            self._get_measure(   0, [0.0, 0.0, 0.0], _prob_zero),
            self._get_measure(_yr1, [4.0, 2.0, 0.0], [5.0, 4.5]),
            self._get_measure(_yr2, [4.0, 2.0, 0.0], [5.0, 4.5]),
        ]

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measures)

        # check results

        # year 0 for the second measure is copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 0)
        assert (pf_to_beta(_pf) == py.approx(_prob_zero[0]))

        # year 20 for the second measure is an interpolated value copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT,_yr1)
        _beta_expected = _prob_zero[0] - _prob_diff * _yr1 / self._end_year
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # year 30 for the third measure is an interpolated value copied from the first measure:
        _pf = _measures[2].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT,_yr2)
        _beta_expected = _prob_zero[0] - _prob_diff * _yr2 / self._end_year
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # all measures are extended with four years:
        _ref = {0, _yr1, _yr1 + 1, _yr2, _yr2 + 1, self._end_year}
        _yrs = _measures[0].mechanism_year_collection.get_years(MeasureTypeEnum.REVETMENT)
        assert _yrs == _ref
        _yrs = _measures[1].mechanism_year_collection.get_years(MeasureTypeEnum.REVETMENT)
        assert _yrs == _ref
        _yrs = _measures[2].mechanism_year_collection.get_years(MeasureTypeEnum.REVETMENT)
        assert _yrs == _ref

    def test_measures_without_year_zero(self):
        # setup
        _measure_list = [self._get_measure(20, [4.0, 2.0, 0.0], [5.0, 4.5])]

        # run test
        with py.raises(ValueError) as exceptionInfo:
            _setyear = SetInvestmentYear()
            _setyear.update_measurelist_with_investment_year(_measure_list)

        # check result
        assert "zero measure not found for this type: REVETMENT" == str(
            exceptionInfo.value
        )
