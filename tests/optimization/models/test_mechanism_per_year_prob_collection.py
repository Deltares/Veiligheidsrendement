import pytest

from vrtool.optimization.models.mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection

class TestMechanismPerYearProbCollection:

    def test_create_collection(self):
        prob = {"overflow": { 0: 0.9, 50: 0.8, 100: 0.7},
                "stability": { 0: 0.85, 50: 0.75, 100: 0.65}}
        collection = MechanismPerYearProbabilityCollection(prob)
        prob50yr = collection.filter("overflow", 50)
        assert prob50yr == 0.8
