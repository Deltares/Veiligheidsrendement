import shutil
from pathlib import Path
from typing import Iterator

import pytest

from tests import get_clean_test_results_dir, test_data
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.exporters.measures.list_of_dict_to_custom_measure_exporter import (
    ListOfDictToCustomMeasureExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.custom_measure import CustomMeasure
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.orm_controllers import initialize_database


class TestListOfDictToCustomMeasureExporter:
    _database_ref_dir = test_data.joinpath("38-1 custom measures")

    def test_initialize_without_db_context_raises_error(self):
        # 1. Define test data.
        _expected_error_mssg = "Database context (SqliteDatabase) required for export."

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            ListOfDictToCustomMeasureExporter(None)

        # 3. Verify expectations.
        assert str(exc_err.value) == _expected_error_mssg

    def _get_db_copy(self, reference_db: Path, request: pytest.FixtureRequest) -> Path:
        # Creates a copy of the database to avoid locking it
        # or corrupting its data.
        _output_directory = get_clean_test_results_dir(request)
        _copy_db = _output_directory.joinpath("vrtool_input.db")
        shutil.copyfile(reference_db, _copy_db)
        assert _copy_db.is_file()

        return _copy_db

    @pytest.fixture(name="exporter_with_valid_db")
    def get_valid_exporter_with_db_context(
        self, request: pytest.FixtureRequest
    ) -> Iterator[ListOfDictToCustomMeasureExporter]:
        # 1. Define test data.
        _db_name = "without_custom_measures.db"
        _db_copy = self._get_db_copy(self._database_ref_dir.joinpath(_db_name), request)

        # Stablish connection
        _test_db_context = initialize_database(_db_copy)

        # Yield item to tests.
        yield ListOfDictToCustomMeasureExporter(_test_db_context)

        # Close connection
        _test_db_context.close()

    def test_initialize_with_db_context(
        self, exporter_with_valid_db: ListOfDictToCustomMeasureExporter
    ):
        # 1. Verify expectations.
        assert isinstance(exporter_with_valid_db, ListOfDictToCustomMeasureExporter)
        assert isinstance(exporter_with_valid_db, OrmExporterProtocol)

    def test_export_dom_without_t0_value(
        self, exporter_with_valid_db: ListOfDictToCustomMeasureExporter
    ):
        # 1. Define test data.
        _custom_measure_dict = dict(
            MEASURE_NAME=MeasureTypeEnum.SOIL_REINFORCEMENT.name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            SECTION_NAME="DummySection",
            TIME=42,
        )
        _list_of_dict = [_custom_measure_dict]
        _expected_error_mssgs = [
            "It was not possible to export the custom measures to the database, detailed error:",
            "Missing t0 beta value for Custom Measure {} - {} - {}".format(
                _custom_measure_dict["MEASURE_NAME"],
                _custom_measure_dict["COMBINABLE_TYPE"],
                _custom_measure_dict["SECTION_NAME"],
            ),
        ]

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert all(
            _error_mssg in str(exc_err.value) for _error_mssg in _expected_error_mssgs
        )

    def test_export_dom_with_t0_value_for_only_one_custom_measure(
        self, exporter_with_valid_db: ListOfDictToCustomMeasureExporter
    ):
        # 1. Define test data.
        _valid_custom_measure_dicts = [
            dict(
                MEASURE_NAME=MeasureTypeEnum.SOIL_REINFORCEMENT.name,
                COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
                SECTION_NAME="DummySection",
                TIME=_t,
            )
            for _t in range(1, 10, 2)
        ]
        _invalid_dict = dict(
            MEASURE_NAME=MeasureTypeEnum.SOIL_REINFORCEMENT.name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            SECTION_NAME="DummySection",
            TIME=42,
        )
        _list_of_dict = _valid_custom_measure_dicts + [_invalid_dict]
        _expected_error_mssg = "It was not possible to export the custom measures to the database, detailed error:\nMissing t0 beta value for Custom Measure {} - {} - {}".format(
            _invalid_dict["MEASURE_NAME"],
            _invalid_dict["COMBINABLE_TYPE"],
            _invalid_dict["SECTION_NAME"],
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert str(exc_err.value) == _expected_error_mssg

    def test_given_only_t0_the_rest_is_constant_over_time(
        self, exporter_with_valid_db: ListOfDictToCustomMeasureExporter
    ):
        # 1. Define test data.
        _selected_mechanism = MechanismEnum.OVERFLOW.name
        _expected_beta = 2.4
        _custom_measure_dict = dict(
            MEASURE_NAME=MeasureTypeEnum.SOIL_REINFORCEMENT.name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            MECHANISM_NAME=_selected_mechanism,
            SECTION_NAME="01A",
            TIME=0,
            BETA=_expected_beta,
            COST=211223,
        )
        _list_of_dict = [_custom_measure_dict]

        # 2. Run test.
        _exported_measures = exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert len(_exported_measures) == 1
        for _em in _exported_measures:
            assert isinstance(_em, CustomMeasure)
            assert _em.beta == _expected_beta
            assert _em.year == 0
            # We should only have one MeasurePerSection,
            # In any case, this is not the test to check said constraint.
            _measure_result = (
                _em.measure.sections_per_measure.get().measure_per_section_result.get()
            )
            _available_t_periods = list(
                x.time
                for x in _em.mechanism.sections_per_mechanism.get()
                .assessment_mechanism_results.select(AssessmentMechanismResult.time)
                .distinct()
            )
            for _t_period in _available_t_periods:
                _found_result = _measure_result.measure_result_mechanisms.where(
                    MeasureResultMechanism.time == _t_period
                ).get_or_none()
                if _found_result is None:
                    pytest.fail(
                        f"No MeasureResultMechanism exported for t = {_t_period}"
                    )
                assert _found_result.beta == _expected_beta
