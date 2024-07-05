from pathlib import Path
from typing import Iterator

import pytest

from tests import test_data


@pytest.fixture(name="valid_version_init_file")
def get_valid_version_init_file_fixture() -> Iterator[Path]:
    _init_file = test_data.joinpath("orm", "version_file.py")
    assert _init_file.exists()
    yield _init_file
