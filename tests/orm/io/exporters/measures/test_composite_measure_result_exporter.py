from vrtool.decision_making.measures.measure_protocol import CompositeMeasureProtocol
from vrtool.orm.io.exporters.measures.composite_measure_result_exporter import (
    CompositeMeasureResultExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class TestCompositeMeasureResultExporter:
    def test_initialize(self):
        _exporter = CompositeMeasureResultExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, CompositeMeasureResultExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_given_valid_composite_measure(self):
        _t_columns = [0, 2, 4, 24, 42]
        _measure_with_params = {}
        _measure_without_params = {}

        class DummyCompositeMeasure(CompositeMeasureProtocol):
            def __init__(self) -> None:
                self.measures = [_measure_with_params, _measure_without_params]

        # 1. Define test data.
        _test_composite_measure = DummyCompositeMeasure()

        # 2. Run test.
        CompositeMeasureResultExporter(None).export_dom(_test_composite_measure)

        # 3. Verify final expectations.
        _expected_measures = len(_t_columns) * len(_test_composite_measure.measures)
        assert len(MeasureResult.select()) == _expected_measures
        assert len(MeasureResultParameter.select()) == _expected_measures * 2
