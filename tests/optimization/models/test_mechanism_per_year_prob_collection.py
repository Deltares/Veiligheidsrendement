import pytest

from vrtool.optimization.models.mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection
from vrtool.common.enums.mechanism_enum import MechanismEnum

class TestMechanismPerYearProbCollection:

    def test_create_collection(self):
        prob = {MechanismEnum.OVERFLOW: { 0: 0.9, 50: 0.8, 100: 0.7},
                MechanismEnum.STABILITY_INNER: { 0: 0.85, 50: 0.75, 100: 0.65}}
        collection = MechanismPerYearProbabilityCollection(prob)
        prob50yr = collection.filter(MechanismEnum.OVERFLOW, 50)
        assert prob50yr == 0.8
