from vrtool.orm.io.exporters.measures.measure_result_exporter import (
    MeasureResultExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class TestMeasureResultExporter:
    def test_initialize(self):
        _exporter = MeasureResultExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureResultExporter)
        assert isinstance(_exporter, OrmExporterProtocol)
