from pathlib import Path

import pytest

from tests import test_results
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.orm_version import OrmVersion

ZERO_VERSION = (0, 0, 0)


class TestOrmVersion:
    def test_initialize(self, valid_version_init_file: Path):
        # 1. Execute test
        _orm_version = OrmVersion(valid_version_init_file)

        # 2. Verify expectations
        assert isinstance(_orm_version, OrmVersion)

    @pytest.mark.parametrize(
        "version_string, expected",
        [
            pytest.param("v1_2_3", (1, 2, 3), id="v1_2_3"),
            pytest.param("1.2.3", (1, 2, 3), id="1.2.3"),
        ],
    )
    def test_parse_version(self, version_string: str, expected: tuple[int, int, int]):
        # 1. Execute test
        _version = OrmVersion.parse_version(version_string)

        # 3. Verify expectations
        assert _version == expected

    def test_construct_version_string(self):
        # 1. Define test data
        _version = (1, 2, 3)

        # 2. Execute test
        _version_string = OrmVersion.construct_version_string(_version)

        # 3. Verify expectations
        assert _version_string == "1.2.3"

    def test_read_version_empty_file_returns_default(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _version_file = test_results.joinpath(request.node.name, "version_file.py")
        _version_file.unlink(missing_ok=True)
        _version_file.parent.mkdir(parents=True, exist_ok=True)
        _version_file.touch(exist_ok=True)
        assert _version_file.exists()

        _orm_version = OrmVersion(_version_file)

        # 2. Execute test
        _version = _orm_version.read_version()

        # 3. Verify expectations
        assert _version == ZERO_VERSION

    def test_read_version(self, valid_version_init_file: Path):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)

        # 2. Execute test
        _version = _orm_version.read_version()

        # 3. Verify expectations
        assert _version == (8, 0, 1)

    def test_get_version(self, valid_version_init_file: Path):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)
        _version = _orm_version.read_version()

        # 2. Execute test
        _version = _orm_version.get_version()

        # 3. Verify expectations
        assert _version == (8, 0, 1)

    def test_set_version(self, valid_version_init_file: Path):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)
        _version = (6, 7, 8)

        # 2. Execute test
        _orm_version.set_version(_version)

        # 3. Verify expectations
        assert _orm_version.get_version() == _version

    @pytest.mark.parametrize(
        "increment_type, expected",
        [
            pytest.param(IncrementTypeEnum.MAJOR, (1, 0, 0), id="major"),
            pytest.param(IncrementTypeEnum.MINOR, (0, 1, 0), id="minor"),
            pytest.param(IncrementTypeEnum.PATCH, (0, 0, 1), id="patch"),
        ],
    )
    def test_add_increment(
        self,
        valid_version_init_file: Path,
        increment_type: IncrementTypeEnum,
        expected: tuple[int, int, int],
    ):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)
        _orm_version.set_version(ZERO_VERSION)

        # 2. Execute test
        _orm_version.add_increment(increment_type)

        # 3. Verify expectations
        assert _orm_version.get_version() == expected

    def test_write_version_existing_file(self, request: pytest.FixtureRequest):
        # 1. Define test data
        _version_file = test_results.joinpath(request.node.name, "version_file.py")
        _version_file.parent.mkdir(parents=True, exist_ok=True)
        _version_file.touch(exist_ok=True)
        assert _version_file.exists()

        _orm_version = OrmVersion(_version_file)
        _version = (1, 2, 3)

        # 2. Execute test
        _orm_version.write_version(_version)

        # 3. Verify expectations
        assert _version_file.exists()
        assert _orm_version.read_version() == _version

    def test_write_version_new_file(self, request: pytest.FixtureRequest):
        # 1. Define test data
        _version_file = test_results.joinpath(request.node.name, "version_file.py")
        _version_file.unlink(missing_ok=True)
        assert not _version_file.exists()

        _orm_version = OrmVersion(_version_file)
        _version = (1, 2, 3)

        # 2. Execute test
        _orm_version.write_version(_version)

        # 3. Verify expectations
        assert _version_file.exists()
        assert _orm_version.read_version() == _version
