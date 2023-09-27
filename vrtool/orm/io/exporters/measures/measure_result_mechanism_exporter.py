from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
import logging
from vrtool.orm.models.measure_result import MeasureResult

from vrtool.orm.models.mechanism_per_section import MechanismPerSection
import pandas as pd


class MeasureResultMechanismExporter(OrmExporterProtocol):
    mechanism_per_section: MechanismPerSection
    measure_result_list: list[MeasureResult]

    def export_dom(self, dom_model: pd.Series) -> None:
        logging.info("Started exporting mechanism {} measure's results.")
        pass
        logging.info("Finished exporting mechanism {} measure's results.")
