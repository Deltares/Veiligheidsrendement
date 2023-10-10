import random
import shutil
from pathlib import Path

import pandas as pd
import pytest
from peewee import SqliteDatabase

import vrtool.orm.models as orm_models
from tests import test_data, test_results
from tests.orm import (
    empty_db_fixture,
    get_basic_combinable_type,
    get_basic_dike_traject_info,
    get_basic_measure_type,
    get_basic_mechanism_per_section,
)
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
    validate_measure_result_export,
)
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    create_basic_optimization_run,
    create_optimization_run_for_selected_measures,
    export_results_measures,
    export_results_optimization,
    export_results_safety_assessment,
    get_dike_section_solutions,
    get_dike_traject,
    get_exported_measure_result_ids,
    initialize_database,
    open_database,
)
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)


class DummyModelsData:
    dike_traject_info = dict(
        traject_name="16-1",
        omega_piping=0.25,
        omega_stability_inner=0.04,
        omega_overflow=0.24,
        a_piping=float("nan"),
        b_piping=300,
        a_stability_inner=0.033,
        b_stability_inner=50,
        beta_max=0.01,
        p_max=0.0001,
        flood_damage=float("nan"),
        traject_length=0.0,
    )
    section_data = dict(
        section_name="section_one",
        dijkpaal_start="start_point",
        dijkpaal_end="end_point",
        meas_start=2.4,
        meas_end=4.2,
        section_length=123,
        in_analysis=True,
        crest_height=1.0,
        annual_crest_decline=2.0,
        cover_layer_thickness=3.0,
        pleistocene_level=4.0,
    )
    mechanism_data = [dict(name="a_mechanism"), dict(name="b_mechanism")]
    buildings_data = [
        dict(distance_from_toe=24, number_of_buildings=2),
        dict(distance_from_toe=42, number_of_buildings=1),
    ]
    characteristic_point_type = ["BIT", "BUT", "BUK", "BIK", "EBL", "BBL"]

    profile_points = [
        dict(x_coordinate=47.0, y_coordinate=5.104),
        dict(x_coordinate=-17.0, y_coordinate=4.996),
        dict(x_coordinate=0, y_coordinate=10.939),
        dict(x_coordinate=3.5, y_coordinate=10.937),
        dict(x_coordinate=42.0, y_coordinate=5.694),
        dict(x_coordinate=25.0, y_coordinate=6.491),
    ]


class TestOrmControllers:
    def test_create_db(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _db_file = test_results / request.node.name / "vrtool_db.db"
        if _db_file.parent.exists():
            shutil.rmtree(_db_file.parent)

        # 2. Run test.
        initialize_database(_db_file)

        # 3. Verify expectations.
        assert _db_file.exists()

    @pytest.mark.skipif(
        condition=(test_data.joinpath("test_db\\vrtool_db.db").exists()),
        reason="Test database already exists. Won't overwrite.",
    )
    def test_create_db_with_data(self):
        # 1. Define database file.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        if _db_file.exists():
            _db_file.unlink()

        initialize_database(_db_file)

        # 2. Define models.
        _dike_traject_info: orm_models.DikeTrajectInfo = (
            orm_models.DikeTrajectInfo.create(**DummyModelsData.dike_traject_info)
        )
        _dike_traject_info.save()

        _dike_section: orm_models.SectionData = orm_models.SectionData.create(
            **(dict(dike_traject=_dike_traject_info) | DummyModelsData.section_data)
        )
        _dike_section.save()

        for _m_dict in DummyModelsData.mechanism_data:
            _mechanism = orm_models.Mechanism.create(**_m_dict)
            _mechanism.save()
            _mechanism_section = orm_models.MechanismPerSection.create(
                mechanism=_mechanism, section=_dike_section
            )
            _mechanism_section.save()

        for _b_dict in DummyModelsData.buildings_data:
            orm_models.Buildings.create(
                **(_b_dict | dict(section_data=_dike_section))
            ).save()

        for idx, _p_point in enumerate(DummyModelsData.profile_points):
            _c_point = orm_models.CharacteristicPointType.create(
                **dict(name=DummyModelsData.characteristic_point_type[idx])
            )
            _c_point.save()
            orm_models.ProfilePoint.create(
                **(
                    _p_point
                    | dict(section_data=_dike_section, profile_point_type=_c_point)
                )
            ).save()

        # 3. Save tables.
        assert _db_file.exists()

    def test_open_database(self):
        # 1. Define test data.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        assert _db_file.is_file()
        _expected_data = DummyModelsData.section_data

        # 2. Run test.
        _sql_db = open_database(_db_file)

        # 3. Verify expectations
        assert isinstance(_sql_db, SqliteDatabase)
        assert any(orm_models.SectionData.select())
        _section_data: orm_models.SectionData = orm_models.SectionData.get_by_id(1)
        assert _section_data.section_name == _expected_data["section_name"]
        assert _section_data.dijkpaal_start == _expected_data["dijkpaal_start"]
        assert _section_data.dijkpaal_end == _expected_data["dijkpaal_end"]
        assert _section_data.meas_start == _expected_data["meas_start"]
        assert _section_data.meas_end == _expected_data["meas_end"]
        assert _section_data.section_length == _expected_data["section_length"]
        assert _section_data.in_analysis == _expected_data["in_analysis"]
        assert _section_data.crest_height == _expected_data["crest_height"]
        assert (
            _section_data.annual_crest_decline == _expected_data["annual_crest_decline"]
        )
        assert (
            _section_data.cover_layer_thickness
            == _expected_data["cover_layer_thickness"]
        )
        assert _section_data.pleistocene_level == _expected_data["pleistocene_level"]

    def test_open_database_when_file_doesnot_exist_raises_value_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _db_file = test_results / request.node.name / "vrtool_db.db"
        assert not _db_file.exists()

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            open_database(_db_file)

        # 3. Verify expectations
        assert str(exc_err.value) == "No file was found at {}".format(_db_file)

    @pytest.fixture
    def database_vrtool_config(self, request: pytest.FixtureRequest) -> VrtoolConfig:
        # 1. Define test data.
        _db_name = "with_valid_data.db"

        _vrtool_config = VrtoolConfig(
            input_directory=(test_data / "test_db"),
            input_database_name=_db_name,
            traject="38-1",
        )
        assert _vrtool_config.input_database_path.is_file()

        _vrtool_config.output_directory = test_results.joinpath(request.node.name)
        if _vrtool_config.output_directory.exists():
            shutil.rmtree(_vrtool_config.output_directory)

        yield _vrtool_config

    def test_get_dike_traject(self, database_vrtool_config: VrtoolConfig):
        # 2. Run test.
        _dike_traject = get_dike_traject(database_vrtool_config)

        # 3. Verify expectations.
        assert isinstance(_dike_traject, DikeTraject)
        assert len(_dike_traject.sections) == 60

        def check_section_reliability(section: DikeSection):
            _recognized_keys = [
                "Overflow",
                "Piping",
                "StabilityInner",
            ]

            def check_key_value(key_value):
                assert (
                    key_value[0] in _recognized_keys
                ), "Mechanism {} not recognized.".format(key_value[0])
                assert isinstance(key_value[1], MechanismReliabilityCollection)

            assert isinstance(section.section_reliability, SectionReliability)
            assert isinstance(section.section_reliability.load, LoadInput)
            assert isinstance(
                section.section_reliability.failure_mechanisms,
                FailureMechanismCollection,
            )

            all(
                map(
                    check_key_value,
                    section.section_reliability.failure_mechanisms._failure_mechanisms.items(),
                )
            )

        assert all(any(_ds.mechanism_data.items()) for _ds in _dike_traject.sections)
        all(map(check_section_reliability, _dike_traject.sections))

    def test_get_dike_section_solutions(self, database_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        database_vrtool_config.T = [0]
        _general_info = DikeTrajectInfo(traject_name="Dummy")
        _dike_section = DikeSection()
        _dike_section.name = "01A"
        _dike_section.Length = 359.0

        # Stability Inner
        _water_load_input = LoadInput([])
        _water_load_input.input["d_cover"] = None
        _water_load_input.input["beta"] = 42.24
        _stability_inner_collection = MechanismReliabilityCollection(
            "StabilityInner", "combinable", database_vrtool_config.T, 2023, 2025
        )
        _stability_inner_collection.Reliability["0"].Input = _water_load_input
        _dike_section.section_reliability.load = _water_load_input
        _dike_section.section_reliability.failure_mechanisms._failure_mechanisms[
            "StabilityInner"
        ] = _stability_inner_collection

        # Initial Geometry
        def _to_record(geom_item: tuple[str, list[int]]) -> dict:
            return dict(type=geom_item[0], x=geom_item[1][0], z=geom_item[1][1])

        _initial_geom = {
            "BUT": [-17.0, 4.996],
            "BUK": [0, 10.939],
            "BIK": [3.5, 10.937],
            "BBL": [25.0, 6.491],
            "EBL": [42.0, 5.694],
            "BIT": [47.0, 5.104],
        }
        _dike_section.InitialGeometry = pd.DataFrame.from_records(
            map(_to_record, _initial_geom.items())
        )
        _dike_section.InitialGeometry.set_index("type", inplace=True, drop=True)

        # Mechanism data
        _dike_section.mechanism_data["StabilityInner"] = [
            ("RW000", "SIMPLE"),
            "combinable",
        ]
        # 2. Run test.
        _solutions = get_dike_section_solutions(
            database_vrtool_config, _dike_section, _general_info
        )

        # 3. Verify expectations.
        assert isinstance(_solutions, Solutions)
        assert any(_solutions.measures)

    @pytest.fixture
    def export_database(self, request: pytest.FixtureRequest) -> SqliteDatabase:
        _db_file = test_data.joinpath("test_db", "empty_db.db")
        _output_dir = test_results.joinpath(request.node.name)
        if _output_dir.exists():
            shutil.rmtree(_output_dir)
        _output_dir.mkdir(parents=True)
        _test_db_file = _output_dir.joinpath("test_db.db")
        shutil.copyfile(_db_file, _test_db_file)

        _connected_db = open_database(_test_db_file)
        _connected_db.close()
        yield _connected_db
        # Make sure it's closed.
        # Perhaps during test something fails and does not get to close
        if isinstance(_connected_db, SqliteDatabase) and not _connected_db.is_closed():
            _connected_db.close()

    def test_export_results_safety_assessment_given_valid_data(
        self, export_database: SqliteDatabase
    ):
        # 1. Define test data.
        export_database.connect()
        _test_mechanism_per_section = get_basic_mechanism_per_section()
        export_database.close()
        _test_section_data = _test_mechanism_per_section.section

        # Dike Section and Dike Traject.
        _reliability_df = pd.DataFrame(
            [4.2, 2.4],
            columns=["42"],
            index=[_test_mechanism_per_section.mechanism.name, "Section"],
        )
        _dummy_section = DikeSection()
        _dummy_section.name = _test_section_data.section_name
        _dummy_section.TrajectInfo = DikeTrajectInfo(
            traject_name=_test_section_data.dike_traject.traject_name
        )
        _dummy_section.section_reliability.SectionReliability = _reliability_df
        _test_traject = DikeTraject()
        _test_traject.sections = [_dummy_section]

        # Safety assessment.
        _safety_assessment = ResultsSafetyAssessment()
        _db_path = Path(export_database.database)
        _safety_assessment.vr_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        _safety_assessment.selected_traject = _test_traject

        assert not any(orm_models.AssessmentSectionResult.select())
        assert not any(orm_models.AssessmentMechanismResult.select())

        # 2. Run test.
        export_results_safety_assessment(_safety_assessment)

        # 3. Verify final expectations.
        assert any(
            orm_models.AssessmentSectionResult.select().where(
                (orm_models.AssessmentSectionResult.section_data == _test_section_data)
                & (orm_models.AssessmentSectionResult.beta == 2.4)
                & (orm_models.AssessmentSectionResult.time == 42)
            )
        )
        assert any(
            orm_models.AssessmentMechanismResult.select().where(
                (
                    orm_models.AssessmentMechanismResult.mechanism_per_section
                    == _test_mechanism_per_section
                )
                & (orm_models.AssessmentMechanismResult.beta == 4.2)
                & (orm_models.AssessmentMechanismResult.time == 42)
            )
        )

    @pytest.fixture
    def results_measures_with_mocked_data(
        self, request: pytest.FixtureRequest, export_database: pytest.FixtureRequest
    ) -> tuple[MeasureResultTestInputData, ResultsMeasures]:
        _measures_input_data = MeasureResultTestInputData.with_measures_type(
            request.param, {}
        )

        # Define vrtool config.
        _database_path = Path(export_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_database_path.parent,
            input_database_name=_database_path.name,
            traject=_measures_input_data.domain_dike_section.TrajectInfo.traject_name,
        )

        # Define solutions.
        _solutions = Solutions(
            dike_section=_measures_input_data.domain_dike_section, config=_vrtool_config
        )
        _solutions.measures = [_measures_input_data.measure]

        # Define results measures object.
        _results_measures = ResultsMeasures()
        _results_measures.vr_config = _vrtool_config
        _results_measures.selected_traject = _measures_input_data
        _results_measures.solutions_dict["sth"] = _solutions

        yield _measures_input_data, _results_measures

    @pytest.mark.parametrize(
        "results_measures_with_mocked_data",
        [
            pytest.param(MeasureWithDictMocked, id="With dictionary"),
            pytest.param(MeasureWithListOfDictMocked, id="With list of dictionaries"),
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
        indirect=True,
    )
    def test_export_results_measures_given_valid_data(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
        export_database: pytest.FixtureRequest,
    ):
        """
        Virtually this test verifies (almost) the same as
        `TestMeasureExporter.test_export_dom_with_valid_data`.
        """
        # 1. Define test data.
        _measures_input_data, _results_measures = results_measures_with_mocked_data
        assert isinstance(_measures_input_data, MeasureResultTestInputData)
        assert isinstance(_results_measures, ResultsMeasures)

        # 2. Run test.
        export_results_measures(_results_measures)

        # 3. Verify expectations.
        validate_measure_result_export(
            _measures_input_data, _measures_input_data.parameters_to_validate
        )

    @pytest.mark.parametrize(
        "results_measures_with_mocked_data",
        [
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
        indirect=True,
    )
    def test_get_selected_measure_result_ids_returns_list_of_exported_results_measures(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
    ):
        # 1. Define test data.
        _measures_input_data, _results_measures = results_measures_with_mocked_data
        assert isinstance(_measures_input_data, MeasureResultTestInputData)
        assert isinstance(_results_measures, ResultsMeasures)
        export_results_measures(_results_measures)

        # 2. Run test.
        _return_values = get_exported_measure_result_ids(_results_measures)

        # 3. Verify expectations.
        assert len(MeasureResult.select()) == 1
        _expected_id = MeasureResult.get().get_id()
        assert _return_values == [_expected_id]

    @pytest.mark.parametrize(
        "results_measures_with_mocked_data",
        [
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
        indirect=True,
    )
    @pytest.mark.skip(reason="Requires mocking a geometry, not worth at the moment.")
    def test_export_optimization_selected_measures_given_valid_data(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
    ):
        # 1. Define test data.
        _measures_input_data, _results_measures = results_measures_with_mocked_data
        assert isinstance(_measures_input_data, MeasureResultTestInputData)
        assert isinstance(_results_measures, ResultsMeasures)
        export_results_measures(_results_measures)
        validate_measure_result_export(
            _measures_input_data, _measures_input_data.parameters_to_validate
        )

        _optimization_run_name = "Test optimization name"

        # 2. Run test.
        _measure_result_selection = len(orm_models.MeasureResult.select()) // 2
        if _measure_result_selection == 0:
            _measure_result_selection = 1
        _measure_result_ids = [
            mr.get_id()
            for mr in orm_models.MeasureResult.select().limit(_measure_result_selection)
        ]
        _return_value = create_optimization_run_for_selected_measures(
            _results_measures.vr_config, _measure_result_ids, _optimization_run_name
        )

        # 3. Verify expectations.
        assert isinstance(_return_value, ResultsMeasures)
        assert len(orm_models.OptimizationType.select()) == len(
            _results_measures.vr_config.design_methods
        )
        for _optimization_type in orm_models.OptimizationType:
            assert isinstance(_optimization_type, orm_models.OptimizationType)

            assert len(_optimization_type.optimization_runs) == 1
            _optimization_run = _optimization_type.optimization_runs[0]

            assert isinstance(_optimization_run, orm_models.OptimizationRun)
            assert _optimization_run.name == _optimization_run_name
            assert (
                _optimization_run.discount_rate
                == _results_measures.vr_config.discount_rate
            )
            assert len(_optimization_run.optimization_run_measure_results) == len(
                _measure_result_ids
            )

    @pytest.mark.parametrize(
        "results_measures_with_mocked_data",
        [
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
        indirect=True,
    )
    def test_create_basic_optimization_run_selects_all_measures(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
    ):
        # 1. Define test data.
        _measures_input_data, _results_measures = results_measures_with_mocked_data
        assert isinstance(_measures_input_data, MeasureResultTestInputData)
        assert isinstance(_results_measures, ResultsMeasures)
        export_results_measures(_results_measures)
        validate_measure_result_export(
            _measures_input_data, _measures_input_data.parameters_to_validate
        )

        _optimization_run_name = "Test optimization name"

        # 2. Run test.
        create_basic_optimization_run(
            _results_measures.vr_config, _optimization_run_name
        )

        # 3. Verify expectations.
        assert len(orm_models.OptimizationType.select()) == len(
            _results_measures.vr_config.design_methods
        )
        for _optimization_type in orm_models.OptimizationType:
            assert isinstance(_optimization_type, orm_models.OptimizationType)

            assert len(_optimization_type.optimization_runs) == 1
            _optimization_run = _optimization_type.optimization_runs[0]

            assert isinstance(_optimization_run, orm_models.OptimizationRun)
            assert _optimization_run.name == _optimization_run_name
            assert (
                _optimization_run.discount_rate
                == _results_measures.vr_config.discount_rate
            )
            assert len(_optimization_run.optimization_run_measure_results) == len(
                orm_models.MeasureResult.select()
            )

    @pytest.mark.parametrize(
        "results_measures_with_mocked_data",
        [
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
        indirect=True,
    )
    @pytest.mark.skip(reason="Work in progress, needs to be completed by VRTOOL-268.")
    def test_export_results_optimization_given_valid_data(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
    ):
        # 1. Define test data.
        _measures_input_data, _results_measures = results_measures_with_mocked_data
        assert isinstance(_measures_input_data, MeasureResultTestInputData)
        assert isinstance(_results_measures, ResultsMeasures)
        export_results_measures(_results_measures)
        validate_measure_result_export(
            _measures_input_data, _measures_input_data.parameters_to_validate
        )

        # Generate default run data.
        _optimization_type = "Test optimization type"
        _test_optimization_type = orm_models.OptimizationType.create(
            name=_optimization_type
        )
        _optimization_run = orm_models.OptimizationRun.create(
            name="PremadeOptimization",
            discount_rate=0.42,
            optimization_type=_test_optimization_type,
        )
        for _measure_result in orm_models.MeasureResult.select():
            orm_models.OptimizationSelectedMeasure.create(
                optimization_run=_optimization_run,
                measure_result=_measure_result,
                investment_year=2023,
            )

        # Define strategies.
        class MockedStrategy(StrategyBase):
            def __init__(self, type, config: VrtoolConfig):
                # First run could just be exporting the index of TakenMeasures.
                self.options = pd.DataFrame(
                    list(map(lambda x: x.id, MeasureResult.select()))
                )  # All possible combinations of MeasureResults (by ID).
                self.options_geotechnical = pd.DataFrame(
                    list(map(lambda x: x.id, MeasureResultMechanism.select()))
                )
                self.options_height = pd.DataFrame(
                    list(map(lambda x: x.id, MeasureResultSection.select()))
                )
                # Measures selected per step
                self.MeasureIndices = pd.DataFrame(
                    list(
                        map(
                            lambda x: [
                                x.id,
                                random.randint(0, len(self.options_geotechnical) - 1),
                                random.randint(0, len(self.options_height) - 1),
                            ],
                            MeasureResult.select(),
                        )
                    )
                )
                # Has a lot of information already present in measure results.
                _measures_columns = [
                    "Section",
                    "option_in",
                    "LCC",
                    "BC",
                    "ID",
                    "name",
                    "yes/no",
                    "dcrest",
                    "dberm",
                    "beta_target",
                    "transition_level",
                ]
                _taken_measure_row = [0] * len(_measures_columns)
                self.TakenMeasures = pd.DataFrame(
                    [_taken_measure_row], columns=_measures_columns
                )  # This is actually OptimizationStep (with extra info).
                _single_existing_measure_result = MeasureResult.select().get()
                self.TakenMeasures["Section"][
                    0
                ] = _single_existing_measure_result.measure_per_section.section.id
                self.TakenMeasures["option_in"][0] = self.options[0][
                    0
                ]  # This actually refers to the `MeasureResult.ID`
                self.TakenMeasures["LCC"][0] = 42.24

        _test_strategy = MockedStrategy(
            type=_optimization_type, config=_results_measures.vr_config
        )

        # Define results optimization object.
        _results_optimization = ResultsOptimization()
        _results_optimization.vr_config = _results_measures.vr_config
        _results_optimization.selected_traject = (
            _measures_input_data.domain_dike_section.TrajectInfo.traject_name
        )
        _results_optimization.results_solutions = _results_measures.solutions_dict
        _results_optimization.results_strategies = [_test_strategy]

        # 2. Run test.
        export_results_optimization(_results_optimization)

        # 3. Verify expectations.
        assert len(orm_models.OptimizationStep.select()) == 1
        assert len(orm_models.OptimizationStepResultMechanism) == len(
            _measures_input_data.t_columns
        )
        assert len(orm_models.OptimizationStepResultSection) == len(
            _measures_input_data.t_columns
        )

        _optimization_step = orm_models.OptimizationStep.get()
        for _t_column in _measures_input_data.t_columns:
            _step_result_mechanism = (
                orm_models.OptimizationStepResultMechanism.get_or_none(
                    optimization_step=_optimization_step, time=_t_column
                )
            )
            assert isinstance(
                _step_result_mechanism, orm_models.OptimizationStepResultMechanism
            )
            _step_result_section = orm_models.OptimizationStepResultSection.get_or_none(
                optimization_step=_optimization_step, time=_t_column
            )
            assert isinstance(
                _step_result_section, orm_models.OptimizationStepResultSection
            )

    def test_clear_assessment_results_clears_all_results(
        self, export_database: SqliteDatabase
    ):
        # Setup
        _db_connection = export_database
        _db_connection.connect()

        assert not any(orm_models.AssessmentSectionResult.select())
        assert not any(orm_models.AssessmentMechanismResult.select())

        traject_info = get_basic_dike_traject_info()

        _mechanisms = [
            self._create_mechanism("mechanism 1"),
            self._create_mechanism("mechanism 2"),
        ]

        self._create_section_with_fully_configured_assessment_results(
            traject_info, "section 1", _mechanisms
        )
        self._create_section_with_fully_configured_assessment_results(
            traject_info, "section 2", _mechanisms
        )

        # Precondition
        assert any(orm_models.AssessmentSectionResult.select())
        assert any(orm_models.AssessmentMechanismResult.select())

        _db_connection.close()

        # Call
        _db_path = Path(_db_connection.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_assessment_results(_vrtool_config)

        # Assert
        _db_connection.connect()

        assert not any(orm_models.AssessmentSectionResult.select())
        assert not any(orm_models.AssessmentMechanismResult.select())

        _db_connection.close()

    def test_clear_measure_result_clears_all_results(
        self, export_database: SqliteDatabase
    ):
        # Setup
        self._generate_measure_results(export_database)

        # Call
        _db_path = Path(export_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_measure_results(_vrtool_config)

        # Assert
        assert not any(orm_models.MeasureResult.select())
        assert not any(orm_models.MeasureResultParameter.select())
        assert not any(orm_models.MeasureResultSection.select())
        assert not any(orm_models.MeasureResultMechanism.select())

    def test_clear_optimization_results_clears_all_results(
        self, export_database: SqliteDatabase
    ):
        # 1. Define test data.
        self._generate_optimization_results(export_database)

        # 2. Run test.
        _db_path = Path(export_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_optimization_results(_vrtool_config)

        # 3. Verify expectations.
        assert not any(orm_models.OptimizationRun.select())
        assert not any(orm_models.OptimizationSelectedMeasure.select())
        assert not any(orm_models.OptimizationStep.select())
        assert not any(orm_models.OptimizationStepResultMechanism.select())
        assert not any(orm_models.OptimizationStepResultSection.select())

    def _generate_measure_results(self, db_connection: SqliteDatabase):
        db_connection.connect()
        traject_info = get_basic_dike_traject_info()

        _measure_type = get_basic_measure_type()
        _combinable_type = get_basic_combinable_type()
        _measures = [
            self._create_measure(_measure_type, _combinable_type, "measure 1"),
            self._create_measure(_measure_type, _combinable_type, "measure 2"),
        ]

        self._create_section_with_fully_configured_measure_results(
            traject_info, "Section 1", _measures
        )
        self._create_section_with_fully_configured_measure_results(
            traject_info, "Section 2", _measures
        )
        db_connection.close()

        assert any(orm_models.MeasureResult.select())
        assert any(orm_models.MeasureResultParameter.select())
        assert any(orm_models.MeasureResultSection.select())
        assert any(orm_models.MeasureResultMechanism.select())

    def _generate_optimization_results(self, db_connection: SqliteDatabase):
        self._generate_measure_results(db_connection)
        if db_connection.is_closed():
            # It could happen it has not been closed.
            db_connection.connect()
        _dummy_optimization_type = orm_models.OptimizationType.create(name="DummyType")
        _optimization_run = orm_models.OptimizationRun.create(
            name="DummyRun",
            discount_rate=0.42,
            optimization_type=_dummy_optimization_type,
        )
        _measure_result = orm_models.MeasureResult.select()[0].get()
        _optimization_selected_measure = orm_models.OptimizationSelectedMeasure.create(
            optimization_run=_optimization_run,
            measure_result=_measure_result,
            investment_year=2021,
        )
        _optimization_step = orm_models.OptimizationStep.create(
            optimization_selected_measure=_optimization_selected_measure, step_number=42
        )
        _mechanism = orm_models.Mechanism.create(name="A Mechanism")
        _mechanism_per_section = orm_models.MechanismPerSection.create(
            mechanism=_mechanism, section=_measure_result.measure_per_section.section
        )
        orm_models.OptimizationStepResultMechanism.create(
            optimization_step=_optimization_step,
            mechanism_per_section=_mechanism_per_section,
            beta=4.2,
            time=20,
            lcc=2023.12,
        )
        orm_models.OptimizationStepResultSection.create(
            optimization_step=_optimization_step,
            beta=4.2,
            time=20,
            lcc=2023.12,
        )

        db_connection.close()

        assert any(orm_models.OptimizationRun.select())
        assert any(orm_models.OptimizationSelectedMeasure.select())
        assert any(orm_models.OptimizationStep.select())
        assert any(orm_models.OptimizationStepResultMechanism.select())
        assert any(orm_models.OptimizationStepResultSection.select())

    def _create_section_with_fully_configured_assessment_results(
        self,
        traject_info: DikeTrajectInfo,
        section_name: str,
        mechanisms: list[orm_models.Mechanism],
    ) -> None:
        section = self._create_basic_section_data(traject_info, section_name)
        self._create_assessment_section_results(section)

        for mechanism in mechanisms:
            mechanism_per_section = self._create_basic_mechanism_per_section(
                section, mechanism
            )
            self._create_assessment_mechanism_results(mechanism_per_section)

    def _create_basic_section_data(
        self, traject_info: DikeTrajectInfo, section_name: str
    ) -> orm_models.SectionData:
        return orm_models.SectionData.create(
            dike_traject=traject_info,
            section_name=section_name,
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )

    def _create_assessment_section_results(
        self, section: orm_models.SectionData
    ) -> None:
        for i in range(2000, 2100, 10):
            orm_models.AssessmentSectionResult.create(
                beta=i / 1000.0, time=i, section_data=section
            )

    def _create_mechanism(self, mechanism_name: str) -> orm_models.Mechanism:
        _mechanism, _ = orm_models.Mechanism.get_or_create(name=mechanism_name)
        return _mechanism

    def _create_basic_mechanism_per_section(
        self, section: orm_models.SectionData, mechanism: orm_models.Mechanism
    ) -> orm_models.MechanismPerSection:
        return orm_models.MechanismPerSection.create(
            section=section, mechanism=mechanism
        )

    def _create_assessment_mechanism_results(
        self, mechanism_per_section: orm_models.MechanismPerSection
    ) -> None:
        for i in range(2000, 2100, 10):
            orm_models.AssessmentMechanismResult.create(
                beta=i / 1000.0, time=i, mechanism_per_section=mechanism_per_section
            )

    def _create_section_with_fully_configured_measure_results(
        self,
        traject_info: DikeTrajectInfo,
        section_name: str,
        measures: list[orm_models.Measure],
    ) -> None:
        section = self._create_basic_section_data(traject_info, section_name)

        _mechanism_per_section = self._create_basic_mechanism_per_section(
            section, self._create_mechanism("TestMechanism")
        )

        for measure in measures:
            measure_per_section = orm_models.MeasurePerSection.create(
                section=section, measure=measure
            )
            self._create_measure_results(measure_per_section, _mechanism_per_section)

    def _create_measure(
        self,
        measure_type: orm_models.MeasureType,
        combinable_type: orm_models.CombinableType,
        measure_name: str,
    ) -> orm_models.Measure:
        return orm_models.Measure.create(
            measure_type=measure_type,
            combinable_type=combinable_type,
            name=measure_name,
            year=20,
        )

    def _create_measure_results(
        self,
        measure_per_section: orm_models.MeasurePerSection,
        mechanism_per_section: MechanismPerSection,
    ) -> None:
        _t_range = list(range(2000, 2100, 10))
        measure_result = orm_models.MeasureResult.create(
            measure_per_section=measure_per_section,
        )
        _measure_result_parameters = self._get_measure_result_parameters(measure_result)
        orm_models.MeasureResultParameter.insert_many(
            _measure_result_parameters
        ).execute()
        orm_models.MeasureResultSection.insert_many(
            self._get_measure_result_section(measure_result, _t_range)
        ).execute()
        orm_models.MeasureResultMechanism.insert_many(
            self._get_measure_result_mechanism(
                measure_result, _t_range, mechanism_per_section
            )
        ).execute()

    def _get_measure_result_section(
        self, measure_result: orm_models.MeasureResult, t_range: list[int]
    ) -> list[dict]:
        cost = 13.37
        for i in t_range:
            yield dict(
                measure_result=measure_result, beta=i / 1000.0, time=i, cost=cost
            )

    def _get_measure_result_mechanism(
        self,
        measure_result: orm_models.MeasureResult,
        t_range: list[int],
        mechanism_per_section: MechanismPerSection,
    ) -> list[dict]:
        for i in t_range:
            yield dict(
                measure_result=measure_result,
                beta=i / 1000.0,
                time=i,
                mechanism_per_section=mechanism_per_section,
            )

    def _get_measure_result_parameters(
        self, measure_result: orm_models.MeasureResult
    ) -> list[dict]:
        for i in range(1, 10):
            yield dict(
                name=f"Parameter {i}", value=i / 10.0, measure_result=measure_result
            )
