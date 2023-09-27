from vrtool.orm.io.exporters.measures.measure_result_collection_exporter import (
    MeasureResultCollectionExporter,
)
from vrtool.orm.io.exporters.measures.measure_type_converters import MeasureDictListAsMeasureResultCollection

from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection



class MeasureDictListExporter(OrmExporterProtocol):
    """
    TODO: Deprecated, can be removed by using the classes
     in `measure_type_converters.py`.
    """
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, measure_dict_list: list) -> None:
        _as_measure_result_collection = MeasureDictListAsMeasureResultCollection(
            measure_dict_list
        )
        MeasureResultCollectionExporter(self._measure_per_section).export_dom(
            _as_measure_result_collection
        )
