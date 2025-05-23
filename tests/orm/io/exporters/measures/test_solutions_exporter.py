from typing import Callable, Type

import pytest

from tests.orm import with_empty_db_context
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
    validate_clean_database,
    validate_measure_result_export,
    validate_no_parameters,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.measures.solutions_exporter import SolutionsExporter
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.section_data import SectionData


class TestSolutionsExporter:
    def test_initialize(self):
        _exporter = SolutionsExporter()

        # Verify expectations.
        assert isinstance(_exporter, SolutionsExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @with_empty_db_context
    def test_get_measure_per_section_given_no_measure_raises_error(self):
        # 1. Define test data.
        _non_existent_id = 42
        assert not any(Measure.select())

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            SolutionsExporter.get_measure_per_section(
                dike_section_name="", traject_name="", measure_id=_non_existent_id
            )

        # 3. Verify expectations.
        _expected_error = "No 'Measure' was found with id: {}.".format(_non_existent_id)
        assert str(exc_err.value) == _expected_error

    @pytest.mark.parametrize(
        "dike_section_name",
        [
            pytest.param("LoremIpsum", id="With section name"),
            pytest.param("", id="Without section name"),
        ],
    )
    @pytest.mark.parametrize(
        "traject_name",
        [
            pytest.param("NisiAliqua", id="With traject name"),
            pytest.param("", id="Without traject name"),
        ],
    )
    @with_empty_db_context
    def test_get_measure_per_section_given_no_section_data_raises_error(
        self,
        dike_section_name: str,
        traject_name: str,
        get_basic_measure_per_section: Callable[[], MeasurePerSection],
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()

        assert not _measure_per_section.section.section_name == dike_section_name
        assert (
            not _measure_per_section.section.dike_traject.traject_name == traject_name
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            SolutionsExporter.get_measure_per_section(
                dike_section_name=dike_section_name,
                traject_name=traject_name,
                measure_id=_measure_per_section.measure.id,
            )

        # 3. Verify expectations.
        _expected_error = (
            "No 'SectionData' was found with name: {}, for 'DikeTraject': {}.".format(
                dike_section_name, traject_name
            )
        )
        assert str(exc_err.value) == _expected_error

    @with_empty_db_context
    def test_get_measure_per_section_given_no_measure_per_section_returns_none(
        self, get_basic_measure_per_section: Callable[[], MeasurePerSection]
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()
        _measure_per_section_id = _measure_per_section.get_id()
        _measure = _measure_per_section.measure
        _section_data = _measure_per_section.section
        _traject = _measure_per_section.section.dike_traject

        # Delete the entry we should be getting with pro
        _measure_per_section.delete_instance()

        assert Measure.get_by_id(_measure.id)
        assert SectionData.get_by_id(_section_data.id)
        assert not any(MeasurePerSection.select())

        # 2. Run test.
        _retrieved_measure_per_section = SolutionsExporter.get_measure_per_section(
            dike_section_name=_section_data.section_name,
            traject_name=_traject.traject_name,
            measure_id=_measure_per_section_id,
        )

        # 3. Verify expectations.
        assert _retrieved_measure_per_section is None

    @with_empty_db_context
    def test_get_measure_per_section_returns_entry_given_valid_arguments(
        self, get_basic_measure_per_section: Callable[[], MeasurePerSection]
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()
        _measure = _measure_per_section.measure
        _section_data = _measure_per_section.section
        _traject = _measure_per_section.section.dike_traject

        # 2. Run test.
        _retrieved_measure_per_section = SolutionsExporter.get_measure_per_section(
            dike_section_name=_section_data.section_name,
            traject_name=_traject.traject_name,
            measure_id=_measure.id,
        )

        # 3. Verify expectations.
        assert _measure_per_section == _retrieved_measure_per_section

    @pytest.mark.parametrize(
        "type_measure",
        [
            pytest.param(MeasureWithDictMocked, id="With dictionary"),
            pytest.param(MeasureWithListOfDictMocked, id="With list of dictionaries"),
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
    )
    @with_empty_db_context
    def test_given_solutions_with_supported_measures_raises_error(
        self, type_measure: Type[MeasureProtocol]
    ):
        # 1. Define test data.
        _measure_parameters = {}
        _measures_test_input_data = MeasureResultTestInputData.with_measures_type(
            type_measure, _measure_parameters
        )
        _exporter = SolutionsExporter()
        _test_solution = Solutions(
            DikeSection(),
            VrtoolConfig(
                traject=_measures_test_input_data.measure_per_section.section.dike_traject.traject_name
            ),
        )
        _test_solution.section_name = (
            _measures_test_input_data.measure_per_section.section.section_name
        )
        _test_solution.measures = [_measures_test_input_data.measure]
        validate_clean_database()
        validate_no_parameters(_measures_test_input_data)

        # 2. Run test.
        _exporter.export_dom(_test_solution)

        # 3. Verify expectations.
        validate_measure_result_export(
            _measures_test_input_data, _measures_test_input_data.parameters_to_validate
        )
