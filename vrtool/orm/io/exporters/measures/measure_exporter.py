from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.decision_making.measures.measure_protocol import (MeasureProtocol, SimpleMeasureProtocol)
from vrtool.decision_making.measures.measure_result_collection_protocol import (MeasureResultProtocol, MeasureResultCollectionProtocol)
from vrtool.orm.io.exporters.measures.simple_measure_exporter import SimpleMeasureExporter
from vrtool.orm.io.exporters.measures.measure_dict_list_exporter import MeasureDictListExporter
from vrtool.orm.io.exporters.measures.measure_result_collection_exporter import MeasureResultCollectionExporter
from vrtool.orm.io.exporters.measures.measure_result_exporter import MeasureResultExporter

class MeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        if isinstance(dom_model, list):
            exporter = MeasureDictListExporter(self._measure_per_section)
            exporter.export_dom(dom_model)
        elif isinstance(dom_model, SimpleMeasureProtocol):
            exporter = SimpleMeasureExporter(self._measure_per_section)
            exporter.export_dom(dom_model)
        elif isinstance(dom_model, MeasureResultCollectionProtocol):
            exporter = MeasureResultCollectionExporter(self._measure_per_section)
            exporter.export_dom(dom_model)
        elif isinstance(dom_model, MeasureResultProtocol):
            exporter = MeasureResultExporter(self._measure_per_section)
            exporter.export_dom(dom_model)
        else:
            raise ValueError("unknown measure type")
