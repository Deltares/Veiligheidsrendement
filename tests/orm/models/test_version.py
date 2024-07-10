from tests.orm import with_empty_db_context
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.version import Version


class TestVersion:
    @with_empty_db_context
    def test_initialize_with_database_fixture(self):
        # 1. Define test data.
        _version_str = "1.2.3"

        # 2. Run test.
        _version = Version.create(orm_version=_version_str)

        # 3. Verify expectations
        assert isinstance(_version, Version)
        assert isinstance(_version, OrmBaseModel)
        assert _version.orm_version == _version_str
