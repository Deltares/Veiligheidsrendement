from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.solutions import Solutions
from vrtool.orm.io.exporters.measures.measure_exporter import MeasureExporter
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
import logging
from vrtool.orm.models import Measure, DikeTrajectInfo, MeasurePerSection, SectionData


class SolutionsExporter(OrmExporterProtocol):
    @staticmethod
    def get_measure_per_section(
        dike_section_name: str, traject_name: str, measure_id: int
    ) -> MeasurePerSection:
        _measure = Measure.get_by_id(measure_id)
        _section_data = (
            SectionData.select()
            .join(DikeTrajectInfo)
            .where(
                (SectionData.section_name == dike_section_name)
                & (DikeTrajectInfo.traject_name == traject_name)
            )
            .get_or_none()
        )
        return (
            MeasurePerSection.select()
            .where(
                (MeasurePerSection.measure == _measure)
                # This ensures we are getting the dike section of the expected dike traject.
                & (MeasurePerSection.section == _section_data)
            )
            .get_or_none()
        )

    def export_dom(self, dom_model: Solutions) -> None:
        _logging_exporting = "exporting solutions for section {}.".format(
            dom_model.section_name
        )
        logging.info("STARTED {}".format(_logging_exporting))
        for measure in dom_model.measures:
            _measure_per_section = self.get_measure_per_section(
                dom_model.section_name,
                dom_model.config.traject,
                measure.parameters["ID"],
            )
            MeasureExporter(_measure_per_section).export_dom(measure)
        logging.info("FINISHED {}".format(_logging_exporting))
        return super().export_dom(dom_model)
