import pytest

from vrtool.common.enums import MechanismEnum
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)


class TestFailureMechanismCollection:
    def test_init_returns_collection_without_mechanisms(self):
        # Call
        collection = FailureMechanismCollection()

        # Assert
        assert len(collection.get_available_mechanisms()) == 0

    def test_given_collection_with_mechanisms_when_getting_available_mechanism_returns_expected_mechanisms(
        self,
    ):
        # Given
        mechanism_one = MechanismEnum.OVERFLOW
        mechanism_two = MechanismEnum.STABILITY_INNER
        mechanism_three = MechanismEnum.PIPING

        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(mechanism_one, "", [], 0, 0)
        )
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(mechanism_two, "", [], 0, 0)
        )
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(mechanism_three, "", [], 0, 0)
        )

        # When
        available_mechanisms = collection.get_available_mechanisms()

        # Then
        assert len(available_mechanisms) == 3
        assert mechanism_one in available_mechanisms
        assert mechanism_two in available_mechanisms
        assert mechanism_three in available_mechanisms

    def test_get_reliability_collection_when_not_in_collection_returns_None(self):
        # Setup
        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(MechanismEnum.OVERFLOW, "", [], 0, 0)
        )

        # Call
        reliability_collection = collection.get_mechanism_reliability_collection(
            MechanismEnum.STABILITY_INNER
        )

        # Assert
        assert reliability_collection is None

    def test_get_reliability_collection_when_in_collection_returns_expected_collection(
        self,
    ):
        # Setup
        mechanism_to_retrieve = MechanismEnum.OVERFLOW
        collection_to_retrieve = MechanismReliabilityCollection(
            mechanism_to_retrieve, "", [], 0, 0
        )

        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(collection_to_retrieve)
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(
                MechanismEnum.STABILITY_INNER, "", [], 0, 0
            )
        )

        # Call
        reliability_collection = collection.get_mechanism_reliability_collection(
            mechanism_to_retrieve
        )

        # Assert
        assert reliability_collection is collection_to_retrieve

    def test_get_all_reliability_collection_when_collection_empty_returns_empty_collection(
        self,
    ):
        # Setup
        collection = FailureMechanismCollection()

        # Call
        reliability_collections = collection.get_all_mechanism_reliability_collections()

        # Assert
        assert len(reliability_collections) == 0

    def test_get_all_reliability_collection_when_collection_not_empty_returns_expected_collection(
        self,
    ):
        # Setup
        reliability_collection_one = MechanismReliabilityCollection(
            MechanismEnum.OVERFLOW, "", [4, 5, 6], 0, 0
        )
        reliability_collection_two = MechanismReliabilityCollection(
            MechanismEnum.STABILITY_INNER, "", [4, 5, 6], 0, 0
        )

        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(
            reliability_collection_one
        )
        collection.add_failure_mechanism_reliability_collection(
            reliability_collection_two
        )

        # Call
        reliability_collections = collection.get_all_mechanism_reliability_collections()

        # Assert
        assert len(reliability_collections) == 2
        assert reliability_collections == [
            reliability_collection_one,
            reliability_collection_two,
        ]

    def test_add_reliability_collection_and_mechanism_not_in_collection_adds_reliability_collection(
        self,
    ):
        # Setup
        mechanism_to_add = MechanismEnum.OVERFLOW
        collection_to_add = MechanismReliabilityCollection(
            mechanism_to_add, "", [], 0, 0
        )
        collection = FailureMechanismCollection()

        # Call
        collection.add_failure_mechanism_reliability_collection(collection_to_add)

        # Assert
        available_mechanisms = collection.get_available_mechanisms()
        assert len(available_mechanisms) == 1
        assert mechanism_to_add in available_mechanisms

        retrieved_collection = collection.get_mechanism_reliability_collection(
            mechanism_to_add
        )
        assert retrieved_collection is collection_to_add

    def test_add_reliability_collection_and_mechanism_in_collection_raises_error(self):
        # Setup
        duplicate_mechanism = MechanismEnum.OVERFLOW
        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(duplicate_mechanism, "", [], 0, 0)
        )

        # Call
        with pytest.raises(ValueError) as exception_error:
            collection.add_failure_mechanism_reliability_collection(
                MechanismReliabilityCollection(duplicate_mechanism, "", [], 0, 0)
            )

        # Assert
        assert (
            str(exception_error.value)
            == f'Mechanism "{duplicate_mechanism}" already added.'
        )

    def test_get_calculation_years_with_empty_collection_returns_empty_collection(self):
        # Setup
        collection = FailureMechanismCollection()

        # Call
        years = collection.get_calculation_years()

        # Assert
        assert len(years) == 0

    def test_get_calculation_years_with_filled_collection_returns_years_of_first_mechanism(
        self,
    ):
        # Setup
        collection = FailureMechanismCollection()
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(
                MechanismEnum.OVERFLOW, "", [4, 5, 6], 0, 0
            )
        )
        collection.add_failure_mechanism_reliability_collection(
            MechanismReliabilityCollection(
                MechanismEnum.STABILITY_INNER, "", [1, 2, 3], 0, 0
            )
        )

        # Call
        years = collection.get_calculation_years()

        # Assert
        assert years == ["4", "5", "6"]
