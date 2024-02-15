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
            MeasureTypeEnum.REVETMENT, 50, beta_to_pf(betas[1])
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
        _measures = [
            self._get_measure( 0, [0.0, 0.0, 0.0], [4.0, 3.5]),
            self._get_measure(20, [4.0, 2.0, 0.0], [5.0, 4.5]),
        ]

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measures)

        # check results

        # year 0 for the second measure is copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 0)
        assert (pf_to_beta(_pf) == py.approx(4.0))

        # year 20 for the second measure is an interpolated value copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 20)
        _beta_expected = 4.0 - 0.5 * 20.0 / 50.0
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # _measure1 is extended with one year:
        assert len(_measures[0].mechanism_year_collection.probabilities) == 3

        # _measure2 is extended with two years:
        assert len(_measures[1].mechanism_year_collection.probabilities) == 4

    def test_three_measures(self):
        # setup
        _measures = [
            self._get_measure( 0, [0.0, 0.0, 0.0], [4.0, 3.5]),
            self._get_measure(20, [4.0, 2.0, 0.0], [5.0, 4.5]),
            self._get_measure(30, [4.0, 2.0, 0.0], [5.0, 4.5]),
        ]

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measures)

        # check results

        # year 0 for the second measure is copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 0)
        assert (pf_to_beta(_pf) == py.approx(4.0))

        # year 20 for the second measure is an interpolated value copied from the first measure:
        _pf = _measures[1].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 20)
        _beta_expected = 4.0 - 0.5 * 20.0 / 50.0
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # year 30 for the third measure is an interpolated value copied from the first measure:
        _pf = _measures[2].mechanism_year_collection.get_probability(MeasureTypeEnum.REVETMENT, 30)
        _beta_expected = 4.0 - 0.5 * 30.0 / 50.0
        assert (pf_to_beta(_pf) == py.approx(_beta_expected))

        # _measure1 is extended with two years (20 and 30):
        assert len(_measures[0].mechanism_year_collection.probabilities) == 4

        # _measure2 is extended with two years (20 and 21):
        assert len(_measures[1].mechanism_year_collection.probabilities) == 4

        # _measure3 is extended with two years (30 and 31):
        assert len(_measures[2].mechanism_year_collection.probabilities) == 4

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
