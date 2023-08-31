from tests.orm import empty_db_fixture, get_basic_measure_per_section
from tests.orm.io.exporters.measures import create_section_reliability
from vrtool.decision_making.measures.measure_protocol import CompositeMeasureProtocol
from vrtool.orm.io.exporters.measures.composite_measure_result_exporter import (
    CompositeMeasureResultExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter
from peewee import SqliteDatabase


class TestCompositeMeasureResultExporter:
    def test_initialize(self):
        _exporter = CompositeMeasureResultExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, CompositeMeasureResultExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_given_valid_composite_measure(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _t_columns = [0, 2, 4, 24, 42]
        _section_reliability = create_section_reliability(_t_columns)
        _measure_with_params = {
            "Cost": 24.42,
            "dcrest": 4.2,
            "dberm": 2.4,
            "Reliability": _section_reliability,
        }
        _measure_without_params = {"Cost": 24.42, "Reliability": _section_reliability}

        class DummyCompositeMeasure(CompositeMeasureProtocol):
            def __init__(self) -> None:
                self.measures = [_measure_with_params, _measure_without_params]

        _test_composite_measure = DummyCompositeMeasure()
        _measure_per_section = get_basic_measure_per_section()

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        CompositeMeasureResultExporter(_measure_per_section).export_dom(
            _test_composite_measure
        )

        # 3. Verify final expectations.
        _expected_measures = len(_t_columns) * len(_test_composite_measure.measures)
        assert len(MeasureResult.select()) == _expected_measures
        assert len(MeasureResultParameter.select()) == len(_t_columns) * 2
