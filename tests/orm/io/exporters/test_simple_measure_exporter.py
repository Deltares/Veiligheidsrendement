from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.simple_measure_exporter import SimpleMeasureExporter


class TestSectionReliabilityExporter:
    def test_initialize(self):
        # Call
        _exporter = SimpleMeasureExporter(None)

        # Assert
        assert isinstance(_exporter, SimpleMeasureExporter)
        assert isinstance(_exporter, OrmExporterProtocol)
