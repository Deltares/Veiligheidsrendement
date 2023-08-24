from tests.orm import empty_db_fixture, get_basic_section_data
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.section_data import SectionData


class TestSectionReliabilityExporter:
    def test_initialize(self):
        # Initialize exporter.
        _exporter = SectionReliabilityExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, SectionReliabilityExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_get_related_section_data(self, empty_db_fixture):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = DikeSection()
        _dike_section.name = _test_section_data.section_name
        _dike_section.TrajectInfo = DikeTrajectInfo(
            _test_section_data.dike_traject.traject_name
        )

        # 2. Run test.
        _related_section_data = SectionReliabilityExporter.get_related_section_data(
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
        _dike_section = DikeSection()
        _dike_section.name = _test_section_data.section_name
        _dike_section.TrajectInfo = DikeTrajectInfo("not_the_traject_in_the_orm")

        # 2. Run test.
        _related_section_data = SectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert _related_section_data is None
