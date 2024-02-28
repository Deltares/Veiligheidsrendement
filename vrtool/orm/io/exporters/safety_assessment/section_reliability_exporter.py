import logging

from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.section_data import SectionData


class SectionReliabilityExporter(OrmExporterProtocol):
    _section_data: SectionData

    def __init__(self, section_data: SectionData) -> None:
        self._section_data = section_data

    def export_dom(self, dom_model: SectionReliability) -> None:
        logging.debug("STARTED exporting Dike Section's reliability (Beta) over time.")
        _added_assessments = []
        for col_name, beta_value in dom_model.SectionReliability.loc["Section"].items():
            _added_assessments.append(
                dict(
                    beta=beta_value,
                    time=int(col_name),
                    section_data=self._section_data,
                )
            )
        AssessmentSectionResult.insert_many(_added_assessments).execute()
        logging.debug("FINISHED exporting Dike Section's reliability (Beta) over time.")
