import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


class TestMechanismPerYearProbCollection:
    def _getMechanismPerYearExample(self):
        _prob = []
        _prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.9))
        _prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.8))
        _prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 100, 0.7))
        _prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.85))
        _prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 50, 0.75))
        _prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 100, 0.65))

        return _prob

    def test_create_collection(self):
        # Setup
        _prob = self._getMechanismPerYearExample()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)

        # Assert
        assert isinstance(_collection, MechanismPerYearProbabilityCollection)

    def test_filter(self):
        # Setup
        _prob = self._getMechanismPerYearExample()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _prob50yr = _collection.filter(MechanismEnum.OVERFLOW, 50)

        # Assert
        assert _prob50yr == 0.8

    def test_interpolation(self):
        # Setup
        _prob = self._getMechanismPerYearExample();

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _collection.add_years([20, 21])

        _prob20yr = _collection.filter(MechanismEnum.OVERFLOW, 20)

        # Assert
        assert _prob20yr == pytest.approx(0.86)
