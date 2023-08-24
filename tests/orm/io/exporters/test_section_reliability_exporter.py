from tests.orm import empty_db_fixture, get_basic_section_data
from tests.orm.io.exporters import section_reliability_with_values
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.assessment_section_results import AssessmentSectionResults
from vrtool.orm.models.section_data import SectionData


class TestSectionReliabilityExporter:
    def test_initialize(self):
        # Initialize exporter.
        _exporter = SectionReliabilityExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, SectionReliabilityExporter)
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

    def test_export_dom_with_valid_arguments(
        self, section_reliability_with_values: SectionReliability, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        assert not any(_test_section_data.assessment_section_results)

        _expected_entries = len(
            section_reliability_with_values.SectionReliability.columns
        )

        # 2. Run test.
        _exporter = SectionReliabilityExporter(_test_section_data)
        _orm_assessments = _exporter.export_dom(section_reliability_with_values)

        # 3. Verify expectations.
        assert len(_orm_assessments) == _expected_entries
        assert len(_test_section_data.assessment_section_results) == _expected_entries
        assert all(
            isinstance(_orm_assessment, AssessmentSectionResults)
            for _orm_assessment in _orm_assessments
        )
        for col_idx, _orm_assessment in enumerate(_orm_assessments):
            assert _orm_assessment.section_data == _test_section_data
            assert (
                _orm_assessment.beta
                == section_reliability_with_values.SectionReliability.loc["Section"][
                    col_idx
                ]
            )
            assert _orm_assessment.time == float(
                section_reliability_with_values.SectionReliability.columns[col_idx]
            )
