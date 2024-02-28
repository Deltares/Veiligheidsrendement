import logging

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.orm.io.exporters.measures.measure_result_exporter import (
    MeasureResultExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection


class MeasureResultCollectionExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureResultCollectionProtocol) -> None:
        logging.debug("STARTED exporting measure's result collection.")
        _measure_result_exporter = MeasureResultExporter(self._measure_per_section)

        for _result in dom_model.result_collection:
            _measure_result_exporter.export_dom(_result)

        logging.debug("FINISHED exporting measure's result collection.")
