from peewee import SqliteDatabase, fn

from tests.orm import empty_db_fixture
from tests.orm.io.exporters.measures import MeasureResultTestInputData
from vrtool.orm.io.exporters.measures.measure_dict_list_exporter import (
    MeasureDictListExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


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
        _input_data = MeasureResultTestInputData()
        _unsupported_param = "unsupported_param"
        _measure_with_params = {
            "id": 42,
            "dcrest": 4.2,
            "dberm": 2.4,
            "Cost": _input_data.expected_cost,
            "Reliability": _input_data.section_reliability,
            _unsupported_param: 13,
        }

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        MeasureDictListExporter(_input_data.measure_per_section).export_dom(
            [_measure_with_params]
        )

        # 3. Verify final expectations.
        assert len(MeasureResult.select()) == len(_input_data.t_columns)
        assert len(MeasureResultParameter.select()) == len(_input_data.t_columns) * 2
        for year in _input_data.t_columns:
            _retrieved_result = MeasureResult.get_or_none(
                (MeasureResult.measure_per_section == _input_data.measure_per_section)
                & (MeasureResult.time == year)
            )

            assert isinstance(_retrieved_result, MeasureResult)
            assert (
                _retrieved_result.beta
                == _input_data.section_reliability.SectionReliability.loc["Section"][
                    year
                ]
            )
            assert _retrieved_result.cost == _input_data.expected_cost
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
            assert not any(
                MeasureResultParameter.select().where(
                    fn.Upper(MeasureResultParameter.name) == _unsupported_param.upper()
                )
            )
