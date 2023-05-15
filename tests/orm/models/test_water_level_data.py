from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.water_level_data import WaterlevelData
from tests.orm.models import empty_db_fixture
import pytest

class TestWaterlevelData:

    @pytest.fixture(autouse=False)
    def valid_section_data(self, empty_db_fixture) -> SectionData:
        # 1. Define test data.
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section= SectionData.create(dike_traject=_test_dike_traject, section_name="TestSection", meas_start=2.4, meas_end=4.2, section_length=123, in_analysis=True, crest_height=24, annual_crest_decline=42)
        yield _test_section

    def test_initialize_with_database_fixture(self, valid_section_data: SectionData):
        # 2. Run test.
        _water_level_data = WaterlevelData.create(section_data=valid_section_data, year=2023, water_level=42, beta=24)

        # 3. Verify expectations.
        assert isinstance(_water_level_data, WaterlevelData)
        assert isinstance(_water_level_data, OrmBaseModel)
        assert _water_level_data.section_data == valid_section_data
        assert _water_level_data in valid_section_data.water_level_data_list
    
    def test_initialize_with_database_fixture_and_water_level_location_id(self, valid_section_data: SectionData):
        # 2. Run test.
        _water_level_data = WaterlevelData.create(section_data=valid_section_data, year=2023, water_level=42, beta=24, water_level_location_id =4)

        # 3. Verify expectations.
        assert isinstance(_water_level_data, WaterlevelData)
        assert isinstance(_water_level_data, OrmBaseModel)
        assert _water_level_data.section_data == valid_section_data
        assert _water_level_data in valid_section_data.water_level_data_list