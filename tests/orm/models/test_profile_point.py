from typing import Callable, Iterator

import pytest

from tests.orm import with_empty_db_context
from vrtool.orm.models.characteristic_point_type import CharacteristicPointType
from vrtool.orm.models.profile_point import ProfilePoint
from vrtool.orm.models.section_data import SectionData


class TestProfilePoint:
    @pytest.fixture(name="profile_point_fixture")
    def _get_profile_proint_fixture(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ) -> Iterator[ProfilePoint]:
        _section_data = get_orm_basic_dike_section()
        _char_point_type = CharacteristicPointType.create(name="TestCharPointType")
        yield ProfilePoint.create(
            profile_point_type=_char_point_type,
            section_data=_section_data,
            x_coordinate=4.2,
            y_coordinate=2.4,
        )

    @with_empty_db_context
    def test_on_delete_characteristic_point_type_cascades(
        self, profile_point_fixture: ProfilePoint
    ):
        # 1. Define test data.
        assert isinstance(profile_point_fixture, ProfilePoint)
        assert any(ProfilePoint.select())

        # 2. Run test.
        CharacteristicPointType.delete_by_id(
            profile_point_fixture.profile_point_type.get_id()
        )

        # 3. Verify expectations.
        assert not any(ProfilePoint)

    @with_empty_db_context
    def test_on_delete_section_data_cascades(self, profile_point_fixture: ProfilePoint):
        # 1. Define test data.
        assert isinstance(profile_point_fixture, ProfilePoint)
        assert any(ProfilePoint.select())

        # 2. Run test.
        SectionData.delete_by_id(profile_point_fixture.section_data.get_id())

        # 3. Verify expectations.
        assert not any(ProfilePoint)
