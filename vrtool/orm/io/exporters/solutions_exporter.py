from vrtool.decision_making.solutions import Solutions
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
import logging


class SolutionsExporter(OrmExporterProtocol):
    def export_dom(self, dom_model: Solutions) -> None:
        _logging_exporting = "exporting solutions for section {}.".format(
            dom_model.section_name
        )
        logging.info("STARTED {}".format(_logging_exporting))
        for measure in dom_model.measures:
            raise NotImplementedError(
                "Exporting measures from solutions is not yet implemented."
            )
        logging.info("FINISHED {}".format(_logging_exporting))
        return super().export_dom(dom_model)
