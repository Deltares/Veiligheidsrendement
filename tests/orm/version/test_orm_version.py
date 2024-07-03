from pathlib import Path

import pytest

from tests import test_results
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.orm_version import OrmVersion


class TestOrmVersion:
    def test_initialize(self, valid_version_init_file: Path):
        # 1. Execute test
        _orm_version = OrmVersion(valid_version_init_file)

        # 2. Verify expectations
        assert isinstance(_orm_version, OrmVersion)

    def test_read_version_no_file_returns_default(self):
        # 1. Define test data
        _orm_version = OrmVersion(None)

        # 2. Execute test
        _version = _orm_version.read_version()

        # 3. Verify expectations
        assert _version == (0, 0, 0)

    def test_read_version_empty_file_returns_default(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _init_file = test_results.joinpath(request.node.name, "version_file.py")
        _init_file.unlink(missing_ok=True)
        _init_file.parent.mkdir(parents=True, exist_ok=True)
        _init_file.touch(exist_ok=True)
        assert _init_file.exists()

        _orm_version = OrmVersion(_init_file)

        # 2. Execute test
        _version = _orm_version.read_version()

        # 3. Verify expectations
        assert _version == (0, 0, 0)

    def test_read_version(self, valid_version_init_file: Path):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)

        # 2. Execute test
        _version = _orm_version.read_version()

        # 3. Verify expectations
        assert _version == (1, 2, 3)

    def test_get_version(self, valid_version_init_file: Path):
        # 1. Define test data
        _orm_version = OrmVersion(valid_version_init_file)
        _version = _orm_version.read_version()

        # 2. Execute test
        _version = _orm_version.get_version()

        # 3. Verify expectations
        assert _version == (1, 2, 3)

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
        _orm_version.set_version((0, 0, 0))

        # 2. Execute test
        _orm_version.add_increment(increment_type)

        # 3. Verify expectations
        assert _orm_version.get_version() == expected

    def test_write_version_existing_file(self, request: pytest.FixtureRequest):
        # 1. Define test data
        _init_file = test_results.joinpath(request.node.name, "version_file.py")
        _init_file.parent.mkdir(parents=True, exist_ok=True)
        _init_file.touch(exist_ok=True)
        assert _init_file.exists()

        _orm_version = OrmVersion(_init_file)
        _orm_version.set_version((1, 2, 3))

        # 2. Execute test
        _orm_version.write_version()

        # 3. Verify expectations
        assert _init_file.exists()
        assert _orm_version.read_version() == (1, 2, 3)

    def test_write_version_new_file(self, request: pytest.FixtureRequest):
        # 1. Define test data
        _init_file = test_results.joinpath(request.node.name, "version_file.py")
        _init_file.unlink(missing_ok=True)
        assert not _init_file.exists()

        _orm_version = OrmVersion(_init_file)
        _orm_version.set_version((1, 2, 3))

        # 2. Execute test
        _orm_version.write_version()

        # 3. Verify expectations
        assert _init_file.exists()
        assert _orm_version.read_version() == (1, 2, 3)
