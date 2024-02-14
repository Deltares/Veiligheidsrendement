import pytest as py

from vrtool.optimization.measures.set_investment_year import SetInvestmentYear
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum


class TestSetInvestmentYear:
    def testTwoMeasures(self):
        # setup
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
        _measure1 = ShMeasure(
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            12,
            0,
            23,
            _collection1,
            4.0,
            2.0,
            0.0,
        )
        _measure2 = ShMeasure(
            MeasureTypeEnum.REVETMENT,
            CombinableTypeEnum.REVETMENT,
            12,
            20,
            23,
            _collection2,
            4.0,
            2.0,
            0.0,
        )
        _measure_list = [_measure1, _measure2]

        # run test
        _setyear = SetInvestmentYear()
        _setyear.update_measurelist_with_investment_year(_measure_list)

        # check result

        # year 0 for _measure2 is copied from _measure1:
        assert (
            _measure2.mechanism_year_collection.get_probability(
                MeasureTypeEnum.REVETMENT, 0
            )
            == 0.8
        )

        # year 20 for _measure2 is an interpolated value copied from _measure1:
        assert _measure2.mechanism_year_collection.get_probability(
            MeasureTypeEnum.REVETMENT, 20
        ) == py.approx(0.76261, abs=1e-5)

        # _measure1 is extended with one year:
        assert len(_measure1.mechanism_year_collection.probabilities) == 3

        # _measure2 is extended with two years:
        assert len(_measure2.mechanism_year_collection.probabilities) == 4
