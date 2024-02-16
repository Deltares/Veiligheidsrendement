import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


class TestMechanismPerYearProbCollection:
    def _get_mechanism_per_year_example(self):
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
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)

        # Assert
        assert isinstance(_collection, MechanismPerYearProbabilityCollection)

    def test_get_probability(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _prob50yr = _collection.get_probability(MechanismEnum.OVERFLOW, 50)

        # Assert
        assert _prob50yr == 0.8

    def test_interpolation(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _collection.add_years([20, 21])

        _prob20yr = _collection.get_probability(MechanismEnum.OVERFLOW, 20)

        # Assert
        assert _prob20yr == pytest.approx(0.86555, abs=1e-5)

    def test_get_probabilities(self):
        # 1. Setup
        _prob = self._get_mechanism_per_year_example()
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _years = 100

        # 2. Call
        _probs = _collection.get_probabilities(
            MechanismEnum.OVERFLOW, list(range(_years))
        )

        # 3. Assert
        assert len(_probs) == _years

    def test_not_adding_existing_year(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _size_before = len(_collection.probabilities)
        _collection.add_years([50])
        _size_after = len(_collection.probabilities)

        # Assert
        assert _size_before == _size_after

    def test_combined_measures(self):
        # Setup
        _collection1 = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection2 = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )

        # Call
        _collection3 = MechanismPerYearProbabilityCollection.combine(
            _collection1, _collection2
        )

        # Assert
        assert len(_collection1.probabilities) == len(_collection3.probabilities)
        assert _collection3.get_probability(
            MechanismEnum.STABILITY_INNER, 0
        ) == pytest.approx(0.9775)
        assert _collection3.get_probability(
            MechanismEnum.STABILITY_INNER, 50
        ) == pytest.approx(0.9375)
        assert _collection3.get_probability(
            MechanismEnum.STABILITY_INNER, 100
        ) == pytest.approx(0.8775)
        assert _collection3.get_probability(MechanismEnum.OVERFLOW, 0) == pytest.approx(
            0.99
        )
        assert _collection3.get_probability(
            MechanismEnum.OVERFLOW, 50
        ) == pytest.approx(0.96)
        assert _collection3.get_probability(
            MechanismEnum.OVERFLOW, 100
        ) == pytest.approx(0.91)

    def test_combined_measures_different_years(self):
        # Setup
        _collection1 = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection2 = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection1.add_years([20])

        # Call
        with pytest.raises(ValueError) as exceptionInfo:
            _collection3 = MechanismPerYearProbabilityCollection.combine(
                _collection1, _collection2
            )

        # Assert
        assert "years not equal in combine" == str(exceptionInfo.value)
