import peewee
import pytest

from tests.orm.models import empty_db_fixture
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestBuildings:
    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # Run test.
        _dike_traject_info = DikeTrajectInfo.create(traject_name="sth")

        # Verify expectations.
        assert isinstance(_dike_traject_info, DikeTrajectInfo)
        assert isinstance(_dike_traject_info, OrmBaseModel)
