from tests.orm import get_basic_section_data, empty_db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.dike_section_reliability_exporter import (
    DikeSectionReliabilityExporter,
)
from vrtool.orm.io.exporters.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.section_data import SectionData


class TestDikeSectionReliabilityExporter:
    def _get_valid_dike_section(
        self, section_name: str, traject_name: str
    ) -> DikeSection:
        _dike_section = DikeSection()
        _dike_section.name = section_name
        _dike_section.TrajectInfo = DikeTrajectInfo(traject_name)
        return _dike_section

    def test_get_related_section_data(self, empty_db_fixture):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )

        # 2. Run test.
        _related_section_data = DikeSectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert isinstance(_related_section_data, SectionData)
        assert _test_section_data == _related_section_data

    def test_get_related_section_data_returns_none_for_different_traject(
        self, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )
        _dike_section.TrajectInfo = DikeTrajectInfo("not_the_traject_in_the_orm")

        # 2. Run test.
        _related_section_data = SectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert _related_section_data is None
