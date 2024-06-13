from typing import Iterable

import pytest

from vrtool.flood_defence_system.dike_traject import DikeTraject


@pytest.fixture(name="mocked_dike_traject")
def get_mocked_dike_traject_fixture() -> Iterable[DikeTraject]:
    class MockedDikeTraject(DikeTraject):
        """A simple dike traject which can be used for testing."""

        def __init__(self) -> None:
            return

    yield MockedDikeTraject()
