from typing import Type
from peewee import SqliteDatabase
import pytest

from tests.orm import empty_db_fixture, get_basic_measure_per_section
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import MeasureResultCollectionProtocol, MeasureResultProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.measures.measure_exporter import MeasureExporter
from vrtool.orm.models.measure_result import MeasureResult
from tests.orm.io.exporters.measures import create_section_reliability
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter

class SimpleMeasureMocked(MeasureProtocol):
    def __init__(self) -> None:
        self.measures = {
            "Cost": 13.37,
            "Reliability": create_section_reliability(list(range(1, 100, 15))),
        }

class ComplexMeasureMocked(MeasureProtocol):
    def __init__(self) -> None:
        self.measures = [{
            "Cost": 13.37,
            "Reliability": create_section_reliability(list(range(1, 100, 15))),
            "id": "my id"
        }]

class MeasureResultMocked(MeasureResultProtocol):
    def __init__(self) -> None:
        self.cost = 13.0
        self.section_reliability = create_section_reliability(list(range(1, 100, 15)))
        self.measure_id = "my id"        

class MeasureResultCollectionMocked(MeasureResultCollectionProtocol):
    def __init__(self) -> None:
        self.result_collection = [MeasureResultMocked()]

class AbcMocked(MeasureProtocol):
    def __init__(self) -> None:
        self.measures = MeasureResultCollectionMocked()


class TestMeasureExporter:

    def test_initialize(self):
        # Call
        _exporter = MeasureExporter(None)

        # Assert
        assert isinstance(_exporter, MeasureExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @pytest.mark.parametrize("type_measure", [pytest.param(SimpleMeasureMocked, id="simple"), pytest.param(ComplexMeasureMocked, id="complex"), pytest.param(AbcMocked, id="measure result object")])
    def test_export_dom_with_valid_data(self, type_measure:Type[MeasureProtocol], empty_db_fixture: SqliteDatabase):
        # Setup
        _measure_per_section = get_basic_measure_per_section()

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        _measure_to_export = type_measure()
        _exporter = MeasureExporter(_measure_per_section)

        # Call
        _exporter.export_dom(_measure_to_export)

        # Assert

        _expected_nr_measure_results = 7
        assert (
            len(_measure_per_section.measure_per_section_result)
            == _expected_nr_measure_results
        )

        assert len(MeasureResult.select()) == _expected_nr_measure_results
