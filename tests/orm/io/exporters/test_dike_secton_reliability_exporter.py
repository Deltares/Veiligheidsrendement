from tests.orm import get_basic_section_data, empty_db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.dike_section_reliability_exporter import (
    DikeSectionReliabilityExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.assessment_mechanism_results import AssessmentMechanismResults
from vrtool.orm.models.assessment_section_results import AssessmentSectionResults
from vrtool.orm.models.section_data import SectionData


class TestDikeSectionReliabilityExporter:
    def test_initialize(self):
        _exporter = DikeSectionReliabilityExporter()

        # Verify expectations.
        assert isinstance(_exporter, DikeSectionReliabilityExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

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

    def test_export_dom_with_valid_data(self):
        # 1. Define test data.
        _exporter = DikeSectionReliabilityExporter()
        _test_section_data = get_basic_section_data()
        _test_dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )
        _expected_assessments = len(
            _test_dike_section.section_reliability.SectionReliability.values
        )

        # 2. Run test.
        _created_assessments = _exporter.export_dom(_test_dike_section)

        # 3. Verify final expectations.
        def filtered(assessment_type) -> list:
            return list(
                filter(lambda x: isinstance(x, assessment_type), _created_assessments)
            )

        assert len(_created_assessments) == _expected_assessments
        assert len(filtered(AssessmentMechanismResults)) == ...
        assert len(filtered(AssessmentSectionResults)) == ...
