from tests.orm import empty_db_fixture, get_basic_section_data
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.water_level_data import WaterlevelData


class TestWaterlevelData:
    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        # 2. Run test.
        _water_level_data = WaterlevelData.create(
            section_data=_test_section_data, year=2023, water_level=42, beta=24
        )

        # 3. Verify expectations.
        assert isinstance(_water_level_data, WaterlevelData)
        assert isinstance(_water_level_data, OrmBaseModel)
        assert _water_level_data.section_data == _test_section_data
        assert _water_level_data in _test_section_data.water_level_data_list

    def test_initialize_with_database_fixture_and_water_level_location_id(
        self, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()

        # 2. Run test.
        _water_level_data = WaterlevelData.create(
            section_data=_test_section_data,
            year=2023,
            water_level=42,
            beta=24,
            water_level_location_id=4,
        )

        # 3. Verify expectations.
        assert isinstance(_water_level_data, WaterlevelData)
        assert isinstance(_water_level_data, OrmBaseModel)
        assert _water_level_data.section_data == _test_section_data
        assert _water_level_data in _test_section_data.water_level_data_list
        assert _water_level_data.water_level_location_id == 4
