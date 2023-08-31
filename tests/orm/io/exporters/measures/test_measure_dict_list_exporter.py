from tests.orm import empty_db_fixture, get_basic_measure_per_section
from tests.orm.io.exporters.measures import create_section_reliability
from vrtool.decision_making.measures.measure_protocol import CompositeMeasureProtocol
from vrtool.orm.io.exporters.measures.measure_dict_list_exporter import (
    MeasureDictListExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter
from peewee import SqliteDatabase


class TestMeasureDictListExporter:
    def test_initialize(self):
        _exporter = MeasureDictListExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureDictListExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_given_valid_composite_measure(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _t_columns = [0, 2, 4, 24, 42]
        _expected_cost = 24.42
        _section_reliability = create_section_reliability(_t_columns)
        _measure_with_params = {
            "Cost": 24.42,
            "dcrest": 4.2,
            "dberm": 2.4,
            "Reliability": _section_reliability,
        }
        _measure_per_section = get_basic_measure_per_section()

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        MeasureDictListExporter(_measure_per_section).export_dom([_measure_with_params])

        # 3. Verify final expectations.
        assert len(MeasureResult.select()) == len(_t_columns)
        assert len(MeasureResultParameter.select()) == len(_t_columns) * 2
        for year in _t_columns:
            _retrieved_result = MeasureResult.get_or_none(
                (MeasureResult.measure_per_section == _measure_per_section)
                & (MeasureResult.time == year)
            )

            assert isinstance(_retrieved_result, MeasureResult)
            assert (
                _retrieved_result.beta
                == _section_reliability.SectionReliability.loc["Section"][year]
            )
            assert _retrieved_result.cost == _expected_cost
            assert len(_retrieved_result.measure_result_parameters) == 2

            def measure_result_parameter_exists(name: str, value: float) -> bool:
                return (
                    MeasureResultParameter.select()
                    .where(
                        (MeasureResultParameter.name == name.upper())
                        & (MeasureResultParameter.value == value)
                        & (MeasureResultParameter.measure_result == _retrieved_result)
                    )
                    .exists()
                )

            assert measure_result_parameter_exists("dcrest", 4.2)
            assert measure_result_parameter_exists("dberm", 2.4)
