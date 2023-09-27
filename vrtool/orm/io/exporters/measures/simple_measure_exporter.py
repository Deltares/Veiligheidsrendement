from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.measures.measure_result_exporter import (
    MeasureResultExporter,
)
from vrtool.orm.io.exporters.measures.measure_type_converters import (
    MeasureDictAsMeasureResult,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection


class SimpleMeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        MeasureResultExporter(self._measure_per_section).export_dom(
            MeasureDictAsMeasureResult(dom_model.measures)
        )
