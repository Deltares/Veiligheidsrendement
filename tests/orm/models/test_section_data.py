from typing import Callable

from tests.orm import with_empty_db_context
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.section_data import SectionData


class TestSectionData:
    @with_empty_db_context
    def test_on_delete_dike_traject_cascades(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ):
        # 1. Define test data.
        assert not any(SectionData.select())
        _section_data = get_orm_basic_dike_section()
        assert any(SectionData.select())

        # 2. Run test.
        DikeTrajectInfo.delete_by_id(_section_data.dike_traject.get_id())

        # 3. Verify expectations.
        assert not any(_section_data.select())

    @with_empty_db_context
    def test_get_flood_damage_value_returns_self_value(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ):
        # 1. Define test data.
        _section_data = get_orm_basic_dike_section()
        _section_data.flood_damage = 2.4
        _section_data.dike_traject.flood_damage = 4.2

        # 2. Run test.
        _flood_damage_value = _section_data.get_flood_damage_value()

        # 3. Verify expectations.
        assert _flood_damage_value == 2.4

    @with_empty_db_context
    def test_get_flood_damage_value_returns_dike_traject_when_null(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ):
        # 1. Define test data.
        _section_data = get_orm_basic_dike_section()
        _section_data.flood_damage = None
        _section_data.dike_traject.flood_damage = 4.2

        # 2. Run test.
        _flood_damage_value = _section_data.get_flood_damage_value()

        # 3. Verify expectations.
        assert _flood_damage_value == 4.2
