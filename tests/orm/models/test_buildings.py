import peewee
import pytest

from tests.orm import get_basic_section_data
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestBuildings:
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture(self):
        # 1. Define test data.
        _test_section = get_basic_section_data()

        # 2. Run test.
        _buildings = Buildings.create(
            section_data=_test_section, distance_from_toe=4.2, number_of_buildings=42
        )

        # 3. Verify expectations
        assert isinstance(_buildings, Buildings)
        assert isinstance(_buildings, OrmBaseModel)
        assert _buildings.distance_from_toe == 4.2
        assert _buildings.number_of_buildings == 42

    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture_no_columns(self):
        # 1. Define test data.
        _test_section = get_basic_section_data()

        # 2. Run test.
        with pytest.raises(peewee.IntegrityError):
            Buildings.create(section_data=_test_section)
