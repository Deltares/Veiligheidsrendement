import pytest

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
        mechanism_name_one = "mechanism 1"
        mechanism_name_two = "mechanism 2"
        mechanism_name_three = "mechanism 3"

        collection = FailureMechanismCollection()
        collection.add_failure_mechanism(
            mechanism_name_one, MechanismReliabilityCollection("", "", [], 0, 0)
        )
        collection.add_failure_mechanism(
            mechanism_name_two, MechanismReliabilityCollection("", "", [], 0, 0)
        )
        collection.add_failure_mechanism(
            mechanism_name_three, MechanismReliabilityCollection("", "", [], 0, 0)
        )

        # When
        available_mechanisms = collection.get_available_mechanisms()

        # Then
        assert len(available_mechanisms) == 3
        assert mechanism_name_one in available_mechanisms
        assert mechanism_name_two in available_mechanisms
        assert mechanism_name_three in available_mechanisms

    def test_get_reliability_collection_when_not_in_collection_returns_None(self):
        # Setup
        collection = FailureMechanismCollection()
        collection.add_failure_mechanism(
            "mechanism", MechanismReliabilityCollection("", "", [], 0, 0)
        )

        # Call
        reliability_collection = collection.get_mechanism_reliability_collection(
            "other mechanism"
        )

        # Assert
        assert reliability_collection is None

    def test_get_reliability_collection_when_in_collection_returns_expected_collection(
        self,
    ):
        # Setup
        mechanism_to_retrieve = "mechanism"
        collection_to_retrieve = MechanismReliabilityCollection("", "", [], 0, 0)

        collection = FailureMechanismCollection()
        collection.add_failure_mechanism(mechanism_to_retrieve, collection_to_retrieve)
        collection.add_failure_mechanism(
            "other mechanism", MechanismReliabilityCollection("", "", [], 0, 0)
        )

        # Call
        reliability_collection = collection.get_mechanism_reliability_collection(
            mechanism_to_retrieve
        )

        # Assert
        assert reliability_collection is collection_to_retrieve

    def test_add_failure_mechanism_and_mechanism_not_in_collection_adds_failure_mechanism(
        self,
    ):
        # Setup
        mechanism_to_add = "mechanism"
        collection_to_add = MechanismReliabilityCollection("", "", [], 0, 0)
        collection = FailureMechanismCollection()

        # Call
        collection.add_failure_mechanism(mechanism_to_add, collection_to_add)

        # Assert
        available_mechanisms = collection.get_available_mechanisms()
        assert len(available_mechanisms) == 1
        assert mechanism_to_add in available_mechanisms

        retrieved_collection = collection.get_mechanism_reliability_collection(
            mechanism_to_add
        )
        assert retrieved_collection is collection_to_add

    def test_add_failure_mechanism_and_mechanism_in_collection_raises_error(self):
        # Setup
        mechanism_to_add = "mechanism"
        collection = FailureMechanismCollection()
        collection.add_failure_mechanism(
            mechanism_to_add, MechanismReliabilityCollection("", "", [], 0, 0)
        )

        # Call
        with pytest.raises(ValueError) as exception_error:
            collection.add_failure_mechanism(
                mechanism_to_add, MechanismReliabilityCollection("", "", [], 0, 0)
            )

        # Assert
        assert (
            str(exception_error.value)
            == f'Mechanism "{mechanism_to_add}" already added.'
        )
