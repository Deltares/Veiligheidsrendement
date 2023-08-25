from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.section_data import SectionData
import logging


class SectionReliabilityExporter(OrmExporterProtocol):
    _section_data: SectionData

    def __init__(self, section_data: SectionData) -> None:
        self._section_data = section_data

    def export_dom(
        self, dom_model: SectionReliability
    ) -> list[AssessmentSectionResult]:
        logging.info("STARTED exporting Dike Section's reliability (Beta) over time.")
        _added_assessments = []
        for col_idx, beta_value in enumerate(
            dom_model.SectionReliability.loc["Section"]
        ):
            _added_assessments.append(
                AssessmentSectionResult.create(
                    beta=beta_value,
                    time=int(dom_model.SectionReliability.columns[col_idx]),
                    section_data=self._section_data,
                )
            )
        logging.info("FINISHED exporting Dike Section's reliability (Beta) over time.")

        return _added_assessments
