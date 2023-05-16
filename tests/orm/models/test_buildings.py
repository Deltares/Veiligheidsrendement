from tests.orm.models import empty_db_fixture
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
import pytest
import peewee

class TestBuildings:

    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # 1. Define test data.
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section= SectionData.create(dike_traject=_test_dike_traject, section_name="TestSection", meas_start=2.4, meas_end=4.2, section_length=123, in_analysis=True, crest_height=24, annual_crest_decline=42)
        
        # 2. Run test.
        _buildings = Buildings.create(section_data=_test_section, distance_from_toe=4.2, number_of_buildings=42)

        # 3. Verify expectations
        assert isinstance(_buildings, Buildings)
        assert isinstance(_buildings, OrmBaseModel)
        assert _buildings.distance_from_toe == 4.2
        assert _buildings.number_of_buildings == 42

    def test_initialize_with_database_fixture_no_columns(self, empty_db_fixture):
        # 1. Define test data.
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section= SectionData.create(dike_traject=_test_dike_traject, section_name="TestSection", meas_start=2.4, meas_end=4.2, section_length=123, in_analysis=True, crest_height=24, annual_crest_decline=42)
        
        # 2. Run test.
        with pytest.raises(peewee.IntegrityError):
            Buildings.create(section_data=_test_section)
