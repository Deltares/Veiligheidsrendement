from typing import Callable

from tests.orm import with_empty_db_context
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.safety_assessment.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.section_data import SectionData


class TestSectionReliabilityExporter:
    def test_initialize(self):
        # Initialize exporter.
        _exporter = SectionReliabilityExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, SectionReliabilityExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @with_empty_db_context
    def test_export_dom_with_valid_arguments(
        self,
        section_reliability_with_values: SectionReliability,
        get_orm_basic_dike_section: Callable[[], SectionData],
    ):
        # 1. Define test data.
        _test_section_data = get_orm_basic_dike_section()
        assert not any(_test_section_data.assessment_section_results)

        _expected_entries = len(
            section_reliability_with_values.SectionReliability.columns
        )

        # 2. Run test.
        SectionReliabilityExporter(_test_section_data).export_dom(
            section_reliability_with_values
        )

        # 3. Verify expectations.
        assert len(AssessmentSectionResult.select()) == _expected_entries
        assert len(_test_section_data.assessment_section_results) == _expected_entries
        for col_idx, _orm_assessment in enumerate(AssessmentSectionResult.select()):
            assert _orm_assessment.section_data == _test_section_data
            assert (
                _orm_assessment.beta
                == section_reliability_with_values.SectionReliability.loc["Section"][
                    col_idx
                ]
            )
            assert _orm_assessment.time == int(
                section_reliability_with_values.SectionReliability.columns[col_idx]
            )
