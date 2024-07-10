import pytest

from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.orm_version import OrmVersion

ZERO_VERSION = (0, 0, 0)


class TestOrmVersion:
    def test_initialize_from_orm(self):
        # 1. Execute test
        _orm_version = OrmVersion.from_orm()

        # 2. Verify expectations
        assert isinstance(_orm_version, OrmVersion)
        assert isinstance(_orm_version.major, int)
        assert isinstance(_orm_version.minor, int)
        assert isinstance(_orm_version.patch, int)

    @pytest.mark.parametrize(
        "increment, expected",
        [
            pytest.param((1, 0, 0), IncrementTypeEnum.MAJOR, id="major"),
            pytest.param((0, 1, 0), IncrementTypeEnum.MINOR, id="minor"),
            pytest.param((0, 0, 1), IncrementTypeEnum.PATCH, id="patch"),
        ],
    )
    def test_get_increment_type(
        self,
        increment: tuple[int, int, int],
        expected: IncrementTypeEnum,
    ):
        # 1. Define test data
        _version_from = OrmVersion(0, 0, 0)
        _version_to = OrmVersion(*increment)

        # 2. Execute test
        _result = OrmVersion.get_increment_type(_version_from, _version_to)

        # 3. Verify expectations
        assert _result == expected
