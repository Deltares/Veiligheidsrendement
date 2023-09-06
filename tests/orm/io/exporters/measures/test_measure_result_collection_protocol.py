from vrtool.orm.io.exporters.measures.measure_result_collection_exporter import (
    MeasureResultCollectionExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class TestMeasureResultCollectionProtocolExporter:
    def test_initialize(self):
        _exporter = MeasureResultCollectionExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureResultCollectionExporter)
        assert isinstance(_exporter, OrmExporterProtocol)
