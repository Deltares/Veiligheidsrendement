import itertools
from operator import itemgetter

import pytest
from peewee import SqliteDatabase

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.exporters.measures.custom_measure_time_beta_calculator import (
    CustomMeasureTimeBetaCalculator,
)
from vrtool.orm.io.exporters.measures.list_of_dict_to_custom_measure_exporter import (
    ListOfDictToCustomMeasureExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.custom_measure import CustomMeasure
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection


class TestListOfDictToCustomMeasureExporter:
    def test_initialize_without_db_context_raises_error(self):
        # 1. Define test data.
        _expected_error_mssg = "Database context (SqliteDatabase) required for export."

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            ListOfDictToCustomMeasureExporter(None)

        # 3. Verify expectations.
        assert str(exc_err.value) == _expected_error_mssg

    @pytest.fixture(name="exporter_with_valid_db")
    def get_valid_custom_measure_exporter_with_db(
        self, custom_measure_db_context: SqliteDatabase
    ):
        yield ListOfDictToCustomMeasureExporter(custom_measure_db_context)

    def test_initialize_with_db_context(
        self, exporter_with_valid_db: ListOfDictToCustomMeasureExporter
    ):
        # 1. Verify expectations.
        assert isinstance(exporter_with_valid_db, ListOfDictToCustomMeasureExporter)
        assert isinstance(exporter_with_valid_db, OrmExporterProtocol)

    @pytest.mark.parametrize(
        "custom_measure_entries",
        [
            pytest.param(
                [dict(TIME=42)],
                id="ONE Measure with ONE Custom Measure WITHOUT t=0",
            ),
            pytest.param(
                [dict(MECHANISM_NAME="MechanismWithoutT0", TIME=42), dict()],
                id="ONE Measure with TWO Custom Measures, ONE WITHOUT t=0",
            ),
            pytest.param(
                [dict(MECHANISM_NAME="MechanismWithoutT0", TIME=42), dict(TIME=24)],
                id="ONE Measure with TWO Custom Measures, BOTH WITHOUT t=0",
            ),
            pytest.param(
                [dict(MEASURE_NAME="MeasureWithoutT0", TIME=42), dict()],
                id="TWO Measures with ONE Custom Measures each, ONE WITHOUT t=0",
            ),
            pytest.param(
                [dict(MEASURE_NAME="MeasureWithoutT0", TIME=42), dict(TIME=24)],
                id="TWO Measures with ONE Custom Measures each, BOTH WITHOUT t=0",
            ),
        ],
    )
    def test_export_dom_without_t0_value_raises(
        self,
        custom_measure_entries: list[dict],
        exporter_with_valid_db: ListOfDictToCustomMeasureExporter,
    ):
        """
        This test mostly targets the inner exception of the protected method
        `_get_grouped_dictionaries_by_measure`. Therefore you may expect some
        concessions or simplifications in the data definition.
        """
        # 1. Define test data.
        _base_custom_measure_dict = dict(
            SECTION_NAME="DummySection",
            MEASURE_NAME=MeasureTypeEnum.SOIL_REINFORCEMENT.name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            MECHANISM_NAME=MechanismEnum.PIPING.name,
            TIME=0,
        )
        _list_of_dict = [
            _base_custom_measure_dict | _de for _de in custom_measure_entries
        ]

        _group_by_custom_measure = itemgetter(
            "MEASURE_NAME", "SECTION_NAME", "MECHANISM_NAME"
        )
        _cm_without_t0 = []
        for _, _cm_grouped_dicts in itertools.groupby(
            sorted(_list_of_dict, key=_group_by_custom_measure),
            key=_group_by_custom_measure,
        ):
            _list_cm_grouped_dicts = list(_cm_grouped_dicts)
            if not any(_cm_dict["TIME"] == 0 for _cm_dict in _list_cm_grouped_dicts):
                _cm_without_t0.append(_list_cm_grouped_dicts[0])

        assert any(
            _cm_without_t0
        ), "All custom measures contain a t0 value, invalid test data."

        def get_custom_measure_error(custom_measure_dict: dict) -> str:
            return "Missing t0 beta value for Custom Measure {} - {} - {} - {}".format(
                custom_measure_dict["MEASURE_NAME"],
                custom_measure_dict["COMBINABLE_TYPE"],
                custom_measure_dict["SECTION_NAME"],
                custom_measure_dict["MECHANISM_NAME"],
            )

        _expected_error_mssg = (
            "It was not possible to export the custom measures to the database, detailed error:\n"
            + "\n".join(get_custom_measure_error(_cm) for _cm in _cm_without_t0)
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert str(exc_err.value) == _expected_error_mssg

    @pytest.mark.parametrize(
        "time_beta_tuples",
        [
            pytest.param([(0, 2.4)], id="ONE custom measure, constant from t=0"),
            pytest.param(
                [(0, 4.2), (27, 2.4)], id="TWO custom measures, constant from t=27"
            ),
        ],
    )
    def test_given_multiple_custom_measures_the_last_is_constant_over_time(
        self,
        time_beta_tuples: list[tuple[int, float]],
        exporter_with_valid_db: ListOfDictToCustomMeasureExporter,
    ):
        # 1. Define test data.
        _known_computation_periods = [0, 19, 20, 25, 50, 75, 100]
        _selected_mechanism = MechanismEnum.OVERFLOW.name
        _measure_cost = 211223
        _custom_measure_base_dict = dict(
            MEASURE_NAME="ROCKS",
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            MECHANISM_NAME=_selected_mechanism,
            SECTION_NAME="01A",
            COST=_measure_cost,
        )
        _list_of_dict = [
            _custom_measure_base_dict | dict(TIME=_time, BETA=_beta)
            for _time, _beta in time_beta_tuples
        ]

        # Define expected values
        _expected_betas = (
            CustomMeasureTimeBetaCalculator.get_interpolated_time_beta_collection(
                time_beta_tuples, _known_computation_periods
            )
        )

        # 2. Run test.
        _exported_measures = exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert len(_exported_measures) == len(time_beta_tuples)
        for _idx, _exported_measure in enumerate(_exported_measures):
            assert isinstance(_exported_measure, CustomMeasure)
            assert _exported_measure.mechanism.name.upper() == _selected_mechanism
            assert _exported_measure.cost == _measure_cost
            _expected_time_beta = time_beta_tuples[_idx]
            assert _exported_measure.year == _expected_time_beta[0]
            assert _exported_measure.beta == _expected_time_beta[1]

        # Get the generated `MeasureResult`
        assert (
            len(set(_em.measure for _em in _exported_measures)) == 1
        ), "Not all exported `Custom Measures` belong to the same `Measure`."

        _measure_result = (
            _exported_measures[0]
            .measure.sections_per_measure.get()
            .measure_per_section_result.get()
        )

        # Verify all created `MeasureResultMechanism` and `MeasureResultSection`
        for _mr_mechanism in _measure_result.measure_result_mechanisms:
            if (
                _mr_mechanism.mechanism_per_section.mechanism.name.upper()
                != _selected_mechanism
            ):
                # When the mechanism was not in our `CustomMeasure` then we expect
                # the value from the assessment.
                _assessment = _mr_mechanism.mechanism_per_section.assessment_mechanism_results.where(
                    AssessmentMechanismResult.time == _mr_mechanism.time
                ).get()
                assert _mr_mechanism.beta == _assessment.beta
            else:
                assert _mr_mechanism.time in _expected_betas
                assert _mr_mechanism.beta == _expected_betas[_mr_mechanism.time]

            _measure_result_section = (
                MeasureResultSection.select()
                .where(
                    (
                        MeasureResultSection.measure_result
                        == _mr_mechanism.measure_result
                    )
                    & (MeasureResultSection.time == _mr_mechanism.time)
                )
                .get()
            )
            assert _measure_result_section.cost == _measure_cost

    def test_export_dom_for_different_sections_creates_different_measures(
        self,
        exporter_with_valid_db: ListOfDictToCustomMeasureExporter,
    ):
        """
        This test mostly targets the inner exception of the protected method
        `_get_grouped_dictionaries_by_measure`. Therefore you may expect some
        concessions or simplifications in the data definition.
        """
        # 1. Define test data.
        _section_names = ["01A", "01B"]
        _measure_name = "ROCKS"
        _base_custom_measure_dict = dict(
            MEASURE_NAME=_measure_name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            MECHANISM_NAME=MechanismEnum.PIPING.name,
            COST=400,
            BETA=6,
            TIME=0,
        )
        _list_of_dict = [
            _base_custom_measure_dict | dict(SECTION_NAME=_s_name)
            for _s_name in _section_names
        ]
        _expected_new_measures = 2

        def measures_with_custom_measure_name() -> list[Measure]:
            return list(Measure.select().where(Measure.name == _measure_name))

        def measures_per_section_with_custom_measure_and_section() -> list[
            MeasurePerSection
        ]:
            return list(
                MeasurePerSection.select().where(
                    (MeasurePerSection.measure.name == _measure_name)
                    & (MeasurePerSection.section.section_name in _section_names)
                )
            )

        assert any(measures_with_custom_measure_name()) is False
        assert any(measures_per_section_with_custom_measure_and_section()) is False

        # 2. Run test.
        _custom_measures = exporter_with_valid_db.export_dom(_list_of_dict)

        # 3. Verify expectations.
        assert len(_custom_measures) == _expected_new_measures

        assert len(measures_with_custom_measure_name()) == _expected_new_measures
        assert (
            len(measures_per_section_with_custom_measure_and_section())
            == _expected_new_measures
        )

        _custom_measures_measures = list(set(_cm.measure for _cm in _custom_measures))
        assert len(_custom_measures_measures) == _expected_new_measures
        for _m in _custom_measures_measures:
            assert _m.name == _measure_name
            assert len(_m.sections_per_measure) == 1
            assert _m.sections_per_measure[0].section.section_name in _section_names
