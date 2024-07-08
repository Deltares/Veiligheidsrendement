from typing import Callable, Iterator

import pytest

from tests.orm import with_empty_db_context
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


class TestMechanismPerSection:
    @pytest.fixture(name="mechanism_per_section_fixture")
    def _get_profile_proint_fixture(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ) -> Iterator[MechanismPerSection]:
        _section_data = get_orm_basic_dike_section()
        _mechanism = Mechanism.create(name="TestMechanism")
        yield MechanismPerSection.create(section=_section_data, mechanism=_mechanism)

    @with_empty_db_context
    def test_on_delete_section_data_cascades(
        self, mechanism_per_section_fixture: MechanismPerSection
    ):
        # 1. Define test data.
        assert isinstance(mechanism_per_section_fixture, MechanismPerSection)
        assert any(MechanismPerSection.select())

        # 2. Run test.
        SectionData.delete_by_id(mechanism_per_section_fixture.section.get_id())

        # 3. Verify expectations.
        assert not any(MechanismPerSection.select())

    @with_empty_db_context
    def test_on_delete_sections_per_mechanism_cascades(
        self, mechanism_per_section_fixture: MechanismPerSection
    ):
        # 1. Define test data.
        assert isinstance(mechanism_per_section_fixture, MechanismPerSection)
        assert any(MechanismPerSection.select())

        # 2. Run test.
        Mechanism.delete_by_id(mechanism_per_section_fixture.mechanism.get_id())

        # 3. Verify expectations.
        assert not any(MechanismPerSection.select())
