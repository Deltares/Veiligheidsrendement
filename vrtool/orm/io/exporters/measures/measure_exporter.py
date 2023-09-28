from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.measures.measure_result_collection_exporter import (
    MeasureResultCollectionExporter,
)
from vrtool.orm.io.exporters.measures.measure_result_type_converter import (
    to_measure_result_collection,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection


class MeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        if not isinstance(dom_model, MeasureProtocol):
            raise TypeError(
                "Unknown measure type: '{}'.".format(type(dom_model).__name__)
            )
        exporter = MeasureResultCollectionExporter(self._measure_per_section)
        exporter.export_dom(to_measure_result_collection(dom_model.measures))
