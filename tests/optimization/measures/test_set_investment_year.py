import pytest as py

from vrtool.optimization.measures.set_investment_year import SetInvestmentYear
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum


class TestSetInvestmentYear:
    def _get_measure_list(self, year: list[int], revetment_params: list[float]) -> list[ShMeasure]:
        _mech_per_year1 = MechanismPerYear(MeasureTypeEnum.REVETMENT, 0, 0.8)
        _mech_per_year2 = MechanismPerYear(MeasureTypeEnum.REVETMENT, 50, 0.7)
        _mech_per_year3 = MechanismPerYear(MeasureTypeEnum.REVETMENT, 0, 0.6)
        _mech_per_year4 = MechanismPerYear(MeasureTypeEnum.REVETMENT, 50, 0.5)
        _collection1 = MechanismPerYearProbabilityCollection(
            [_mech_per_year1, _mech_per_year2]
        )
        _collection2 = MechanismPerYearProbabilityCollection(
            [_mech_per_year3, _mech_per_year4]
        )
        _dummy_cost = 999.0
        _dummy_lcc  = 999.9
        _measure1 = ShMeasure(
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            _dummy_cost,
            year[0],
            _dummy_lcc,
            _collection1,
            revetment_params[0],
            revetment_params[1],
            revetment_params[2],
        )
        _measure2 = ShMeasure(
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            _dummy_cost,
            year[1],
            _dummy_lcc,
            _collection2,
            revetment_params[0],
            revetment_params[1],
            revetment_params[2],
        )
        return [_measure1, _measure2]

    def test_two_measures(self):
        # setup
        _measure_list = self._get_measure_list([0, 20], [4.0, 2.0, 0.0])

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measure_list)

        # check result

        # year 0 for _measure2 is copied from _measure1:
        assert (
            _measure_list[1].mechanism_year_collection.get_probability(
                MeasureTypeEnum.REVETMENT, 0
            )
            == 0.8
        )

        # year 20 for _measure2 is an interpolated value copied from _measure1:
        assert _measure_list[1].mechanism_year_collection.get_probability(
            MeasureTypeEnum.REVETMENT, 20
        ) == py.approx(0.76261, abs=1e-5)

        # _measure1 is extended with one year:
        assert len(_measure_list[0].mechanism_year_collection.probabilities) == 3

        # _measure2 is extended with two years:
        assert len(_measure_list[1].mechanism_year_collection.probabilities) == 4

    def test_four_measures(self):
        # setup
        _measure_list = self._get_measure_list([0, 20], [4.0, 2.0, 0.0])
        _measure_list.extend(self._get_measure_list([0, 30], [3.5, 2.1, 0.0]))

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measure_list)

        # check result

        # year 0 for _measure2 is copied from _measure1:
        assert (
            _measure_list[1].mechanism_year_collection.get_probability(
                MeasureTypeEnum.REVETMENT, 0
            )
            == 0.8
        )

        # year 20 for _measure2 is an interpolated value copied from _measure1:
        assert _measure_list[1].mechanism_year_collection.get_probability(
            MeasureTypeEnum.REVETMENT, 20
        ) == py.approx(0.76261, abs=1e-5)

        # year 30 for _measure4 is an interpolated value copied from _measure1:
        assert _measure_list[3].mechanism_year_collection.get_probability(
            MeasureTypeEnum.REVETMENT, 30
        ) == py.approx(0.74257, abs=1e-5)

        # _measure1 is extended with one year:
        assert len(_measure_list[0].mechanism_year_collection.probabilities) == 3

        # _measure2 is extended with two years:
        assert len(_measure_list[1].mechanism_year_collection.probabilities) == 4

        # _measure3 is extended with one year:
        assert len(_measure_list[2].mechanism_year_collection.probabilities) == 3

        # _measure4 is extended with two years:
        assert len(_measure_list[3].mechanism_year_collection.probabilities) == 4

    def test_measures_without_year_zero(self):
        # setup
        _measure_list = self._get_measure_list([20, 30], [4.0, 2.0, 0.0])

        # run test
        with py.raises(ValueError) as exceptionInfo:
            _setyear = SetInvestmentYear()
            _setyear.update_measurelist_with_investment_year(_measure_list)

        # check result
        assert "equal measure for year=0 not found" == str(exceptionInfo.value)
