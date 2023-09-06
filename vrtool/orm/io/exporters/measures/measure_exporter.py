from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import MeasureResultCollectionProtocol
from vrtool.orm.io.exporters.measures.simple_measure_exporter import SimpleMeasureExporter
from vrtool.orm.io.exporters.measures.measure_dict_list_exporter import MeasureDictListExporter
from vrtool.orm.io.exporters.measures.measure_result_collection_exporter import MeasureResultCollectionExporter

class MeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        if isinstance(dom_model.measures, list):
            exporter = MeasureDictListExporter(self._measure_per_section)
            exporter.export_dom(dom_model.measures)
        elif isinstance(dom_model.measures, dict):
            exporter = SimpleMeasureExporter(self._measure_per_section)
            exporter.export_dom(dom_model)
        elif isinstance(dom_model.measures, MeasureResultCollectionProtocol):
            exporter = MeasureResultCollectionExporter(self._measure_per_section)
            exporter.export_dom(dom_model.measures)
        else:
            raise ValueError(f"unknown measure type: {type(dom_model)}")
