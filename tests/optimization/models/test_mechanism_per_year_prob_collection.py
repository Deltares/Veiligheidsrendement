import pytest

from vrtool.optimization.models.mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection
from vrtool.optimization.models.mechanism_per_year import MechanismPerYear
from vrtool.common.enums.mechanism_enum import MechanismEnum

class TestMechanismPerYearProbCollection:

    def test_create_collection(self):
        prob = []
        prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.9))
        prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.8))
        prob.append(MechanismPerYear(MechanismEnum.OVERFLOW, 100, 0.7))
        prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.85))
        prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 50, 0.75))
        prob.append(MechanismPerYear(MechanismEnum.STABILITY_INNER, 100, 0.65))
        collection = MechanismPerYearProbabilityCollection(prob)
        prob50yr = collection.filter(MechanismEnum.OVERFLOW, 50)
        assert prob50yr == 0.8
