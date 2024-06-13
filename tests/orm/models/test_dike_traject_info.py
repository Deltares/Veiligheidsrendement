import pytest

from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestBuildings:
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture(self):
        # Run test.
        _dike_traject_info = DikeTrajectInfo.create(traject_name="sth")

        # Verify expectations.
        assert isinstance(_dike_traject_info, DikeTrajectInfo)
        assert isinstance(_dike_traject_info, OrmBaseModel)
