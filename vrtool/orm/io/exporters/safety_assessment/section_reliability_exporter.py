import logging

from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.section_data import SectionData
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class SectionReliabilityExporter(OrmExporterProtocol):
    _section_data: SectionData

    def __init__(self, section_data: SectionData) -> None:
        self._section_data = section_data

    def export_dom(self, dom_model: SectionReliability) -> None:
        logging.debug("STARTED exporting Dike Section's reliability (Beta) over time.")
        _added_assessments = []
        for _year, _pf in dom_model.get_reliabilities().items():
            _added_assessments.append(
                dict(
                    beta=pf_to_beta(_pf),
                    time=_year,
                    section_data=self._section_data,
                )
            )
        AssessmentSectionResult.insert_many(_added_assessments).execute()
        logging.debug("FINISHED exporting Dike Section's reliability (Beta) over time.")
