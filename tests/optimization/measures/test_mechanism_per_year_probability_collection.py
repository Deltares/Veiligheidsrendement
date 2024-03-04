import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class TestMechanismPerYearProbabilityCollection:
    def _get_mechanism_per_year_example(
        self, delta_mechanism: [MechanismEnum] = [MechanismEnum.PIPING], delta: float = 0.0
    ) -> list[MechanismPerYear]:
        _prob = []
        for mechanism in [MechanismEnum.PIPING, MechanismEnum.OVERFLOW, MechanismEnum.STABILITY_INNER]:
            if mechanism in delta_mechanism:
                _prob.append(MechanismPerYear(mechanism, 0, 0.5 + delta))
                _prob.append(MechanismPerYear(mechanism, 50, 0.6 + delta))
                _prob.append(MechanismPerYear(mechanism, 100, 0.7 + delta))
            else:
                _prob.append(MechanismPerYear(mechanism, 0, 0.4))
                _prob.append(MechanismPerYear(mechanism, 50, 0.5))
                _prob.append(MechanismPerYear(mechanism, 100, 0.6))

        return _prob

    def test_create_collection(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)

        # Assert
        assert isinstance(_collection, MechanismPerYearProbabilityCollection)

    def test_get_probability_existing(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _prob50yr = _collection.get_probability(MechanismEnum.OVERFLOW, 50)

        # Assert
        assert _prob50yr == 0.5

    def test_get_probability_non_existing_raises_error(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)

        # Assert
        with pytest.raises(StopIteration) as exception_error:
            _prob50yr = _collection.get_probability(MechanismEnum.OVERFLOW, 77)

        assert str(exception_error.value) == ""

    def test_interpolation(self):
        # Setup
        _prob = self._get_mechanism_per_year_example()

        # Call
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _collection.add_years([20, 21])

        _prob20yr = _collection.get_probability(MechanismEnum.OVERFLOW, 20)

        # Assert
        assert _prob20yr == pytest.approx(0.43959, abs=1e-5)

    def test_beta_existing(self):
        # 1. Define test data
        _probs = self._get_mechanism_per_year_example()
        _collection = MechanismPerYearProbabilityCollection(_probs)
        _mech = MechanismEnum.OVERFLOW
        _prob_exp = _collection.get_probability(_mech, 50)
        _beta_exp = pf_to_beta(_prob_exp)

        # 2. Run test
        _beta = _collection.get_beta(_mech, 50)

        # 3. Verify expectations
        assert _beta == pytest.approx(_beta_exp)

    def test_get_beta_non_existing_raises_error(self):
        # 1. Define test data
        _probs = self._get_mechanism_per_year_example()
        _collection = MechanismPerYearProbabilityCollection(_probs)

        # 2. Run test
        with pytest.raises(StopIteration) as exception_error:
            _beta = _collection.get_beta(MechanismEnum.OVERFLOW, 77)

        # 3. Verify expectations
        assert str(exception_error.value) == ""

    def test_get_probabilities(self):
        # 1. Setup
        _prob = self._get_mechanism_per_year_example()
        _collection = MechanismPerYearProbabilityCollection(_prob)
        _years = 101

        # 2. Call
        _probs = _collection.get_probabilities(
            MechanismEnum.OVERFLOW, list(range(_years))
        )

        # 3. Assert
        assert len(_probs) == _years
        assert _probs[0] == pytest.approx(
            [
                p.probability
                for p in _collection.probabilities
                if p.year == 0 and p.mechanism == MechanismEnum.OVERFLOW
            ][0],
        )
        assert _probs[-1] == pytest.approx(
            [
                p.probability
                for p in _collection.probabilities
                if p.year == 100 and p.mechanism == MechanismEnum.OVERFLOW
            ][0],
        )

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
        _collection_prim = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection_sec = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example(delta = 0.076)
        )
        _collection_init = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )

        # Call
        _collection_res = MechanismPerYearProbabilityCollection.combine(
            _collection_prim, _collection_sec, _collection_init
        )

        # Assert
        assert len(_collection_prim.probabilities) == len(_collection_res.probabilities)
        assert _collection_res.get_probability(
            MechanismEnum.STABILITY_INNER, 0
        ) == pytest.approx(0.4)
        assert _collection_res.get_probability(
            MechanismEnum.STABILITY_INNER, 50
        ) == pytest.approx(0.5)
        assert _collection_res.get_probability(
            MechanismEnum.STABILITY_INNER, 100
        ) == pytest.approx(0.6)
        assert _collection_res.get_probability(
            MechanismEnum.PIPING, 0
        ) == pytest.approx(0.576)
        assert _collection_res.get_probability(
            MechanismEnum.PIPING, 50
        ) == pytest.approx(0.676)
        assert _collection_res.get_probability(
            MechanismEnum.PIPING, 100
        ) == pytest.approx(0.776)

    def test_combined_measures_different_years(self):
        # Setup
        _collection_prim = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection_sec = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example()
        )
        _collection_init = MechanismPerYearProbabilityCollection(
            self._get_mechanism_per_year_example(delta = 0.0)
        )
        _collection_prim.add_years([20])

        # Call
        with pytest.raises(ValueError) as exceptionInfo:
            _collection_res = MechanismPerYearProbabilityCollection.combine(
                _collection_prim, _collection_sec, _collection_init
            )

        # Assert
        assert "years not equal in combine" == str(exceptionInfo.value)
