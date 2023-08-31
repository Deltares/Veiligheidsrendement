import pandas as pd
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture, get_basic_measure_per_section
from vrtool.decision_making.measures.measure_protocol import SimpleMeasureProtocol
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.simple_measure_exporter import SimpleMeasureExporter
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class TestSimpleMeasureExporter:
    class MeasureTest(SimpleMeasureProtocol):
        def __init__(self) -> None:
            self.measures = {
                "Cost": 13.37,
                "Reliability": self._create_section_reliability(),
            }

        def _create_section_reliability(self) -> SectionReliability:
            _section_reliability = SectionReliability()

            _years = list(range(1, 100, 15))
            _section_reliability.SectionReliability = pd.DataFrame.from_dict(
                {
                    "IrrelevantMechanism1": [year / 12.0 for year in _years],
                    "IrrelevantMechanism2": [year / 13.0 for year in _years],
                    "Section": [year / 10.0 for year in _years],
                },
                orient="index",
                columns=_years,
            )
            return _section_reliability

    def test_initialize(self):
        # Call
        _exporter = SimpleMeasureExporter(None)

        # Assert
        assert isinstance(_exporter, SimpleMeasureExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_with_valid_data(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _measure_per_section = get_basic_measure_per_section()

        assert not any(_measure_per_section.measure_per_section_result)

        _measure_to_export = self.MeasureTest()
        _exporter = SimpleMeasureExporter(_measure_per_section)

        # Call
        _exporter.export_dom(_measure_to_export)

        # Assert

        _reliability_to_export: SectionReliability = _measure_to_export.measures[
            "Reliability"
        ]
        _row_to_export = _reliability_to_export.SectionReliability.loc["Section"]
        _expected_nr_measure_results = len(_row_to_export)
        assert (
            len(_measure_per_section.measure_per_section_result)
            == _expected_nr_measure_results
        )

        assert len(MeasureResult.select()) == _expected_nr_measure_results
        assert not any(MeasureResultParameter.select())

        _expected_cost = _measure_to_export.measures["Cost"]
        for year in _row_to_export.index:
            _retrieved_result = MeasureResult.get_or_none(
                (MeasureResult.measure_per_section == _measure_per_section)
                & (MeasureResult.time == year)
            )

            assert isinstance(_retrieved_result, MeasureResult)
            assert _retrieved_result.beta == _row_to_export[year]
            assert _retrieved_result.cost == _expected_cost
            assert not any(_retrieved_result.measure_result_parameters)
