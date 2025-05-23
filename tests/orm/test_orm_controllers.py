import itertools
import shutil
from operator import itemgetter
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from peewee import SqliteDatabase, fn

import vrtool.orm.models as orm
from tests import (
    get_copy_of_reference_directory,
    get_vrtool_config_test_copy,
    test_data,
    test_externals,
    test_results,
)
from tests.optimization.conftest import (  # These imports are required by `test_export_results_optimization_given_valid_data`
    _get_section_with_combinations,
    _get_section_with_measures,
)
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
    validate_measure_result_export,
)
from vrtool.api import ApiRunWorkflows
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.greedy_strategy import GreedyStrategy
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategies.strategy_step import StrategyStep
from vrtool.decision_making.strategies.target_reliability_strategy import (
    TargetReliabilityStrategy,
)
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
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm import __version__
from vrtool.orm.io.exporters.measures.custom_measure_time_beta_calculator import (
    CustomMeasureTimeBetaCalculator,
)
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.orm_controllers import (
    add_custom_measures,
    brute_clear_custom_measure,
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    create_optimization_run_for_selected_measures,
    export_results_measures,
    export_results_optimization,
    export_results_safety_assessment,
    get_all_measure_results_with_supported_investment_years,
    get_dike_section_solutions,
    get_dike_traject,
    get_exported_measure_result_ids,
    import_results_measures_for_optimization,
    initialize_database,
    open_database,
    safe_clear_custom_measure,
)
from vrtool.orm.version.migration.database_version import DatabaseVersion
from vrtool.orm.version.orm_version import OrmVersion
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
    mechanism_data = [
        MechanismEnum.OVERFLOW.legacy_name,
        MechanismEnum.STABILITY_INNER.legacy_name,
    ]
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
        assert DatabaseVersion.from_database(_db_file) == OrmVersion.from_orm()

    @pytest.mark.skipif(
        condition=(test_data.joinpath("test_db", "vrtool_db.db").exists()),
        reason="Test database already exists. Won't overwrite.",
    )
    def test_create_db_with_data(self):
        # 1. Define database file.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        if _db_file.exists():
            _db_file.unlink()

        initialize_database(_db_file)

        # 2. Define models.
        _dike_traject_info: orm.DikeTrajectInfo = orm.DikeTrajectInfo.create(
            **DummyModelsData.dike_traject_info
        )
        _dike_traject_info.save()

        _dike_section: orm.SectionData = orm.SectionData.create(
            **(dict(dike_traject=_dike_traject_info) | DummyModelsData.section_data)
        )
        _dike_section.save()

        for _mech_name in DummyModelsData.mechanism_data:
            _mech_inst = orm.Mechanism.create(name=_mech_name)
            _mech_inst.save()
            _mechanism_section = orm.MechanismPerSection.create(
                mechanism=_mech_inst, section=_dike_section
            )
            _mechanism_section.save()

        for _b_dict in DummyModelsData.buildings_data:
            orm.Buildings.create(**(_b_dict | dict(section_data=_dike_section))).save()

        for idx, _p_point in enumerate(DummyModelsData.profile_points):
            _c_point = orm.CharacteristicPointType.create(
                **dict(name=DummyModelsData.characteristic_point_type[idx])
            )
            _c_point.save()
            orm.ProfilePoint.create(
                **(
                    _p_point
                    | dict(section_data=_dike_section, profile_point_type=_c_point)
                )
            ).save()

        orm.Version.create(orm_version=__version__).save()

        # 3. Save tables.
        assert _db_file.exists()

    def test_open_database(self):
        # 1. Define test data.
        _db_file = test_data.joinpath("test_db", "vrtool_db.db")
        assert _db_file.is_file()
        _expected_data = DummyModelsData.section_data

        # 2. Run test.
        _sql_db = open_database(_db_file)

        # 3. Verify expectations
        assert isinstance(_sql_db, SqliteDatabase)
        assert any(orm.SectionData.select())
        _section_data: orm.SectionData = orm.SectionData.get_by_id(1)
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

    def test_open_database_with_incompatible_version_raises_value_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _db_file = test_data.joinpath("test_db", "empty_db.db")
        _output_dir = test_results.joinpath(request.node.name)
        if _output_dir.exists():
            shutil.rmtree(_output_dir)
        _output_dir.mkdir(parents=True)
        _test_db_file = _output_dir.joinpath("test_db.db")
        shutil.copyfile(_db_file, _test_db_file)

        _db_version = DatabaseVersion.from_database(_test_db_file)
        _db_version.major += 1
        _db_version.update_version(_db_version)

        # 2. Run test
        with pytest.raises(ValueError) as exc_err:
            open_database(_test_db_file)

        # 3. Verify expectations
        assert (
            str(exc_err.value)
            == "Database ORM version does not match the current ORM version."
        )

    def test_open_database_when_file_doesnot_exist_raises_value_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _db_file = test_results.joinpath(request.node.name, "vrtool_db.db")
        assert not _db_file.exists()

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            open_database(_db_file)

        # 3. Verify expectations
        assert str(exc_err.value) == "No file was found at {}".format(_db_file)

    @pytest.fixture(name="database_vrtool_config")
    def _get_database_vrtool_config_fixture(
        self, request: pytest.FixtureRequest
    ) -> Iterator[VrtoolConfig]:
        # 1. Define test data.
        _test_db = test_data.joinpath("test_db", "with_valid_data.db")

        _output_directory = test_results.joinpath(request.node.name)
        if _output_directory.exists():
            shutil.rmtree(_output_directory)

        # Generate a custom `VrtoolConfig`
        _vrtool_config = VrtoolConfig(
            input_directory=_test_db.parent,
            input_database_name=_test_db.name,
            traject="38-1",
            output_directory=_output_directory,
        )
        assert _vrtool_config.input_database_path.is_file()

        yield _vrtool_config

    def test_get_dike_traject(self, database_vrtool_config: VrtoolConfig):
        # 2. Run test.
        _dike_traject = get_dike_traject(database_vrtool_config)

        # 3. Verify expectations.
        assert isinstance(_dike_traject, DikeTraject)
        assert len(_dike_traject.sections) == 60

        def check_section_reliability(section: DikeSection):
            _recognized_keys = [
                MechanismEnum.OVERFLOW,
                MechanismEnum.PIPING,
                MechanismEnum.STABILITY_INNER,
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
        _dike_section.cover_layer_thickness = 42
        _dike_section.sensitive_fraction_piping = 1
        _dike_section.sensitive_fraction_stability_inner = 1
        _dike_section.TrajectInfo = _general_info
        _general_info.aPiping = 1
        _general_info.aStabilityInner = 1

        # Stability Inner
        _water_load_input = LoadInput([])
        _water_load_input.input["d_cover"] = None
        _water_load_input.input["beta"] = np.array([42.24])
        _water_load_input.input["piping_reduction_factor"] = 1000
        _stability_inner_collection = MechanismReliabilityCollection(
            MechanismEnum.STABILITY_INNER,
            ComputationTypeEnum.SIMPLE,
            database_vrtool_config.T,
            2023,
            2025,
        )
        _stability_inner_collection.Reliability["0"].Input = _water_load_input
        _dike_section.section_reliability.load = _water_load_input
        _dike_section.section_reliability.failure_mechanisms._failure_mechanisms[
            MechanismEnum.STABILITY_INNER
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
        _dike_section.mechanism_data[MechanismEnum.STABILITY_INNER] = [
            ("RW000", ComputationTypeEnum.SIMPLE)
        ]

        # 2. Run test.
        _solutions = get_dike_section_solutions(
            database_vrtool_config, _dike_section, _general_info
        )

        # 3. Verify expectations.
        assert isinstance(_solutions, Solutions)
        assert any(_solutions.measures)

    def test_export_results_safety_assessment_given_valid_data(
        self,
        persisted_database: SqliteDatabase,
        get_basic_mechanism_per_section: Callable[[], MechanismPerSection],
    ):
        # 1. Define test data.
        persisted_database.connect()
        _test_mechanism_per_section = get_basic_mechanism_per_section()
        persisted_database.close()
        _test_section_data = _test_mechanism_per_section.section

        # Dike Section and Dike Traject.
        _mech_name = MechanismEnum.get_enum(
            _test_mechanism_per_section.mechanism_name
        ).name
        _reliability_df = pd.DataFrame(
            [4.2, 2.4],
            columns=["42"],
            index=[_mech_name, "Section"],
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
        _db_path = Path(persisted_database.database)
        _safety_assessment.vr_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        _safety_assessment.selected_traject = _test_traject

        assert not any(orm.AssessmentSectionResult.select())
        assert not any(orm.AssessmentMechanismResult.select())

        # 2. Run test.
        export_results_safety_assessment(_safety_assessment)

        # 3. Verify final expectations.
        assert any(
            orm.AssessmentSectionResult.select().where(
                (orm.AssessmentSectionResult.section_data == _test_section_data)
                & (orm.AssessmentSectionResult.beta == 2.4)
                & (orm.AssessmentSectionResult.time == 42)
            )
        )
        assert any(
            orm.AssessmentMechanismResult.select().where(
                (
                    orm.AssessmentMechanismResult.mechanism_per_section
                    == _test_mechanism_per_section
                )
                & (orm.AssessmentMechanismResult.beta == 4.2)
                & (orm.AssessmentMechanismResult.time == 42)
            )
        )

    @pytest.fixture(name="results_measures_with_mocked_data")
    def _get_results_measures_with_mocked_data_fixture(
        self, request: pytest.FixtureRequest, persisted_database: pytest.FixtureRequest
    ) -> Iterator[tuple[MeasureResultTestInputData, ResultsMeasures]]:
        _measures_input_data = MeasureResultTestInputData.with_measures_type(
            request.param, {}
        )

        # Define vrtool config.
        _database_path = Path(persisted_database.database)
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
        _measure_result_selection = len(orm.MeasureResult.select()) // 2
        if _measure_result_selection == 0:
            _measure_result_selection = 1
        _measure_result_ids = [
            mr.get_id()
            for mr in orm.MeasureResult.select().limit(_measure_result_selection)
        ]
        _return_value = create_optimization_run_for_selected_measures(
            _results_measures.vr_config, _optimization_run_name, _measure_result_ids
        )

        # 3. Verify expectations.
        assert isinstance(_return_value, dict)
        assert len(orm.OptimizationType.select()) == len(
            _results_measures.vr_config.design_methods
        )
        for _optimization_type in orm.OptimizationType:
            assert isinstance(_optimization_type, orm.OptimizationType)

            assert len(_optimization_type.optimization_runs) == 1
            _optimization_run = _optimization_type.optimization_runs[0]

            assert isinstance(_optimization_run, orm.OptimizationRun)
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
    def test_export_results_optimization_given_valid_data(
        self,
        results_measures_with_mocked_data: tuple[
            MeasureResultTestInputData, ResultsMeasures
        ],
        section_with_combinations: SectionAsInput,
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
        _test_optimization_type = orm.OptimizationType.create(name=_optimization_type)
        _optimization_run = orm.OptimizationRun.create(
            name="PremadeOptimization",
            discount_rate=0.42,
            optimization_type=_test_optimization_type,
        )
        for _measure_result in orm.MeasureResult.select():
            orm.OptimizationSelectedMeasure.create(
                optimization_run=_optimization_run,
                measure_result=_measure_result,
                investment_year=0,
            )

        # Define strategies.
        class MockedStrategy(StrategyProtocol):
            optimization_steps: list[StrategyStep]

            def __init__(self):
                self.sections = [section_with_combinations]
                _aggregated_measure_combination = AggregatedMeasureCombination(
                    sh_combination=self.sections[0].sh_combinations[1],
                    sg_combination=self.sections[0].sg_combinations[0],
                    measure_result_id=1,
                    year=0,
                )
                _section_idx = 0
                self.sections[_section_idx].aggregated_measure_combinations = [
                    _aggregated_measure_combination
                ]

                self.optimization_steps = [
                    StrategyStep(
                        step_number=1,
                        section_idx=_section_idx,
                        measure=(_section_idx, 1, 1),
                        aggregated_measure=_aggregated_measure_combination,
                        probabilities={
                            MechanismEnum.STABILITY_INNER: np.linspace(
                                0.1, 0.6, 100
                            ).reshape((100, 1)),
                            MechanismEnum.OVERFLOW: np.linspace(0.01, 0.1, 100).reshape(
                                (100, 1)
                            ),
                        },
                        total_risk=100.0,
                        total_cost=84.0,
                    )
                ]
                self.time_periods = [0, 2, 4, 24, 42]

        _test_strategy = MockedStrategy()

        # Define results optimization object.
        _results_optimization = ResultsOptimization()
        _results_optimization.vr_config = _results_measures.vr_config
        _results_optimization.selected_traject = (
            _measures_input_data.domain_dike_section.TrajectInfo.traject_name
        )
        _results_optimization.results_strategies = [_test_strategy]

        # 2. Run test.
        export_results_optimization(_results_optimization, [_optimization_run.id])

        # 3. Verify expectations.
        assert len(orm.OptimizationStepResultMechanism) == 10
        assert len(orm.OptimizationStepResultSection) == 5
        assert len(orm.OptimizationStep.select()) == 1
        _optimization_step = orm.OptimizationStep.get()
        assert _optimization_step.total_lcc == 84.0
        assert _optimization_step.total_risk == 100.0

    def test_clear_assessment_results_clears_all_results(
        self,
        persisted_database: SqliteDatabase,
        get_orm_basic_dike_traject_info: Callable[[], DikeTrajectInfo],
    ):
        # Setup
        _db_connection = persisted_database
        _db_connection.connect()

        assert not any(orm.AssessmentSectionResult.select())
        assert not any(orm.AssessmentMechanismResult.select())

        traject_info = get_orm_basic_dike_traject_info()

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
        assert any(orm.AssessmentSectionResult.select())
        assert any(orm.AssessmentMechanismResult.select())

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

        assert not any(orm.AssessmentSectionResult.select())
        assert not any(orm.AssessmentMechanismResult.select())

        _db_connection.close()

    def test_clear_measure_result_clears_all_results(
        self,
        persisted_database: SqliteDatabase,
        generate_optimization_results: Callable[[SqliteDatabase, str], None],
    ):
        # Setup
        generate_optimization_results(persisted_database, "TestMeasureType")

        # Call
        _db_path = Path(persisted_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_measure_results(_vrtool_config)

        # Assert
        assert not any(orm.MeasureResult.select())
        assert not any(orm.MeasureResultParameter.select())
        assert not any(orm.MeasureResultSection.select())
        assert not any(orm.MeasureResultMechanism.select())
        assert not any(orm.OptimizationRun.select())
        assert not any(orm.OptimizationSelectedMeasure.select())
        assert not any(orm.OptimizationStep.select())
        assert not any(orm.OptimizationStepResultMechanism.select())
        assert not any(orm.OptimizationStepResultSection.select())

    def test_clear_measure_result_does_not_clear_custom_results(
        self,
        persisted_database: SqliteDatabase,
        generate_optimization_results: Callable[[SqliteDatabase, str], None],
    ):
        # Setup
        assert not any(orm.MeasureResult.select())
        generate_optimization_results(persisted_database, "Custom")

        # Call
        _db_path = Path(persisted_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_measure_results(_vrtool_config)

        # Assert
        # - Custom results should still be present.
        assert any(orm.MeasureResult.select())
        assert any(orm.MeasureResultParameter.select())
        assert any(orm.MeasureResultSection.select())
        assert any(orm.MeasureResultMechanism.select())
        # - All optimization results should be cleared.
        assert not any(orm.OptimizationRun.select())
        assert not any(orm.OptimizationSelectedMeasure.select())
        assert not any(orm.OptimizationStep.select())
        assert not any(orm.OptimizationStepResultMechanism.select())
        assert not any(orm.OptimizationStepResultSection.select())

    def test_clear_optimization_results_clears_all_results(
        self,
        persisted_database: SqliteDatabase,
        generate_optimization_results: Callable[[SqliteDatabase, str], None],
    ):
        # 1. Define test data.
        generate_optimization_results(persisted_database, "TestMeasureType")

        # 2. Run test.
        _db_path = Path(persisted_database.database)
        _vrtool_config = VrtoolConfig(
            input_directory=_db_path.parent,
            input_database_name=_db_path.name,
        )
        clear_optimization_results(_vrtool_config)

        # 3. Verify expectations.
        assert not any(orm.OptimizationRun.select())
        assert not any(orm.OptimizationSelectedMeasure.select())
        assert not any(orm.OptimizationStep.select())
        assert not any(orm.OptimizationStepResultMechanism.select())
        assert not any(orm.OptimizationStepResultSection.select())

    @pytest.fixture(name="generate_measure_results")
    def _generate_measure_results_fixture(
        self,
        get_orm_basic_dike_traject_info: Callable[[], orm.DikeTrajectInfo],
        get_basic_measure_type: Callable[[str], MeasureType],
        get_basic_combinable_type: Callable[[], CombinableType],
    ) -> Iterator[Callable[[SqliteDatabase, str], None]]:
        def generate_measure_results(
            db_connection: SqliteDatabase, measure_type_name: str
        ) -> None:
            if db_connection.is_closed():
                db_connection.connect()
            traject_info = get_orm_basic_dike_traject_info()

            _measure_type = get_basic_measure_type(measure_type_name)
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

            assert any(orm.MeasureResult.select())
            assert any(orm.MeasureResultParameter.select())
            assert any(orm.MeasureResultSection.select())
            assert any(orm.MeasureResultMechanism.select())

        yield generate_measure_results

    @pytest.fixture(name="generate_optimization_results")
    def _generate_optimization_results_fixture(
        self,
        generate_measure_results: Callable[[SqliteDatabase, str], None],
    ) -> Iterator[Callable[[SqliteDatabase, str], None]]:
        def generate_optimization_results(
            db_connection: SqliteDatabase, measure_type_name: str
        ) -> None:
            generate_measure_results(db_connection, measure_type_name)
            if db_connection.is_closed():
                # It could happen it has not been closed.
                db_connection.connect()
            _dummy_optimization_type = orm.OptimizationType.create(name="DummyType")
            _optimization_run = orm.OptimizationRun.create(
                name="DummyRun",
                discount_rate=0.42,
                optimization_type=_dummy_optimization_type,
            )
            _measure_result = orm.MeasureResult.select()[0].get()
            _optimization_selected_measure = orm.OptimizationSelectedMeasure.create(
                optimization_run=_optimization_run,
                measure_result=_measure_result,
                investment_year=2021,
            )
            _optimization_step = orm.OptimizationStep.create(
                optimization_selected_measure=_optimization_selected_measure,
                step_number=42,
            )
            _mechanism = orm.Mechanism.create(name="A Mechanism")
            _mechanism_per_section = orm.MechanismPerSection.create(
                mechanism=_mechanism,
                section=_measure_result.measure_per_section.section,
            )
            orm.OptimizationStepResultMechanism.create(
                optimization_step=_optimization_step,
                mechanism_per_section=_mechanism_per_section,
                beta=4.2,
                time=20,
                lcc=2023.12,
            )
            orm.OptimizationStepResultSection.create(
                optimization_step=_optimization_step,
                beta=4.2,
                time=20,
                lcc=2023.12,
            )

            db_connection.close()

            assert any(orm.OptimizationRun.select())
            assert any(orm.OptimizationSelectedMeasure.select())
            assert any(orm.OptimizationStep.select())
            assert any(orm.OptimizationStepResultMechanism.select())
            assert any(orm.OptimizationStepResultSection.select())

        yield generate_optimization_results

    def _create_section_with_fully_configured_assessment_results(
        self,
        traject_info: DikeTrajectInfo,
        section_name: str,
        mechanisms: list[orm.Mechanism],
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
    ) -> orm.SectionData:
        return orm.SectionData.create(
            dike_traject=traject_info,
            section_name=section_name,
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )

    def _create_assessment_section_results(self, section: orm.SectionData) -> None:
        for i in range(2000, 2100, 10):
            orm.AssessmentSectionResult.create(
                beta=i / 1000.0, time=i, section_data=section
            )

    def _create_mechanism(self, mechanism_name: str) -> orm.Mechanism:
        _mech_inst, _ = orm.Mechanism.get_or_create(name=mechanism_name)
        return _mech_inst

    def _create_basic_mechanism_per_section(
        self, section: orm.SectionData, mech_inst: orm.Mechanism
    ) -> orm.MechanismPerSection:
        return orm.MechanismPerSection.create(section=section, mechanism=mech_inst)

    def _create_assessment_mechanism_results(
        self, mechanism_per_section: orm.MechanismPerSection
    ) -> None:
        for i in range(2000, 2100, 10):
            orm.AssessmentMechanismResult.create(
                beta=i / 1000.0, time=i, mechanism_per_section=mechanism_per_section
            )

    def _create_section_with_fully_configured_measure_results(
        self,
        traject_info: DikeTrajectInfo,
        section_name: str,
        measures: list[orm.Measure],
    ) -> None:
        section = self._create_basic_section_data(traject_info, section_name)

        _mechanism_per_section = self._create_basic_mechanism_per_section(
            section, self._create_mechanism(MechanismEnum.OVERFLOW.name)
        )

        for measure in measures:
            measure_per_section = orm.MeasurePerSection.create(
                section=section, measure=measure
            )
            self._create_measure_results(measure_per_section, _mechanism_per_section)

    def _create_measure(
        self,
        measure_type: orm.MeasureType,
        combinable_type: orm.CombinableType,
        measure_name: str,
    ) -> orm.Measure:
        return orm.Measure.create(
            measure_type=measure_type,
            combinable_type=combinable_type,
            name=measure_name,
        )

    def _create_measure_results(
        self,
        measure_per_section: orm.MeasurePerSection,
        mechanism_per_section: MechanismPerSection,
    ) -> None:
        _t_range = list(range(2000, 2100, 10))
        measure_result = orm.MeasureResult.create(
            measure_per_section=measure_per_section,
        )
        _measure_result_parameters = self._get_measure_result_parameters(measure_result)
        orm.MeasureResultParameter.insert_many(_measure_result_parameters).execute()
        orm.MeasureResultSection.insert_many(
            self._get_measure_result_section(measure_result, _t_range)
        ).execute()
        orm.MeasureResultMechanism.insert_many(
            self._get_measure_result_mechanism(
                measure_result, _t_range, mechanism_per_section
            )
        ).execute()

    def _get_measure_result_section(
        self, measure_result: orm.MeasureResult, t_range: list[int]
    ) -> Iterator[dict]:
        cost = 13.37
        for i in t_range:
            yield dict(
                measure_result=measure_result, beta=i / 1000.0, time=i, cost=cost
            )

    def _get_measure_result_mechanism(
        self,
        measure_result: orm.MeasureResult,
        t_range: list[int],
        mechanism_per_section: MechanismPerSection,
    ) -> Iterator[dict]:
        for i in t_range:
            yield dict(
                measure_result=measure_result,
                beta=i / 1000.0,
                time=i,
                mechanism_per_section=mechanism_per_section,
            )

    def _get_measure_result_parameters(
        self, measure_result: orm.MeasureResult
    ) -> Iterator[dict]:
        for i in range(1, 10):
            yield dict(
                name=f"Parameter {i}", value=i / 10.0, measure_result=measure_result
            )

    def test_import_results_measures_for_optimization_given_valid_case(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_dir_name = str(Path("reported_bugs", "test_stability_multiple_scenarios"))
        _test_case_dir = get_copy_of_reference_directory(_test_dir_name)

        _vrtool_config = get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        with open_database(_vrtool_config.input_database_path).connection_context():
            _measures_to_import = [(omr.id, 0) for omr in MeasureResult.select()]
            _imported_data = import_results_measures_for_optimization(
                _vrtool_config, _measures_to_import
            )

        # 3. Verify final expectations.
        assert any(_imported_data)
        assert all(
            (
                isinstance(_imp_data, SectionAsInput)
                and any(_imp_data.initial_assessment.probabilities)
            )
            for _imp_data in _imported_data
        )


class TestCustomMeasureDetail:
    """
    This test class mostly covers integration tests for `CustomMeasureDetail` workflows.
    """

    _custom_measures_test_dir = test_data.joinpath("38-1_custom_measures")

    def _get_custom_measure_dict(
        self,
        measure_name: str,
        measure_section: str,
        measure_mechanism: MechanismEnum,
        measure_year: int,
        measure_cost: float,
        measure_beta: float,
    ) -> dict:
        return dict(
            MEASURE_NAME=measure_name,
            COMBINABLE_TYPE=CombinableTypeEnum.FULL.name,
            SECTION_NAME=measure_section,
            MECHANISM_NAME=measure_mechanism.name,
            TIME=measure_year,
            COST=measure_cost,
            BETA=measure_beta,
        )

    @pytest.mark.parametrize(
        "custom_measure_dict_list",
        [
            pytest.param(
                [
                    {
                        "MEASURE_NAME": "ROCKS",
                        "SECTION_NAME": "01A",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MECHANISM_NAME": MechanismEnum.PIPING.name,
                        "COST": 50.0,
                        "TIME": _t,
                        "BETA": _beta,
                    }
                    for (_t, _beta) in zip(
                        [0, 19, 20, 25, 50, 75, 100], np.linspace(7, 4, num=7)
                    )
                ],
                id="MVP test, measure with all required Time",
            ),
            pytest.param(
                [
                    {
                        "SECTION_NAME": "01A",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MEASURE_NAME": "ROCKS",
                        "COST": 50.0,
                        "MECHANISM_NAME": _mechanism_enum.name,
                        "TIME": _time,
                        "BETA": _beta,
                    }
                    for _time, _beta, _mechanism_enum in [
                        (0, 4.2, MechanismEnum.OVERFLOW),
                        (50, 2.4, MechanismEnum.OVERFLOW),
                        (0, 4.2, MechanismEnum.PIPING),
                    ]
                ]
                + [
                    {
                        "SECTION_NAME": "01A",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MEASURE_NAME": "TREES",
                        "MECHANISM_NAME": MechanismEnum.OVERFLOW.name,
                        "TIME": 0,
                        "COST": 23.12,
                        "BETA": 3.0,
                    },
                ],
                id="Integration test",
            ),
            pytest.param(
                [
                    {
                        "MEASURE_NAME": "rocky 2",
                        "SECTION_NAME": "01A",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MECHANISM_NAME": MechanismEnum.OVERFLOW.name,
                        "COST": 1000,
                        "BETA": 6.6,
                        "TIME": _time,
                    }
                    for _time in [0, 40]
                ],
                id="Workflow 1: SAME measure, ONLY DIFFERENT time NOT present IN ASSESSMENT",
            ),
            pytest.param(
                [
                    {
                        "SECTION_NAME": _section_name,
                        "MEASURE_NAME": "rocky 2",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MECHANISM_NAME": MechanismEnum.OVERFLOW.name,
                        "TIME": 0,
                        "COST": 1000,
                        "BETA": 6.6,
                    }
                    for _section_name in ["01A", "01B"]
                ],
                id="Workflow 2a: SAME measure, ONLY DIFFERENT section",
            ),
            pytest.param(
                [
                    {
                        "MEASURE_NAME": "rocky 2",
                        "COMBINABLE_TYPE": CombinableTypeEnum.FULL.name,
                        "MECHANISM_NAME": MechanismEnum.OVERFLOW.name,
                        "TIME": 0,
                        "SECTION_NAME": _section_name,
                        "COST": _cost,
                        "BETA": _beta,
                    }
                    for _section_name, _cost, _beta in [
                        ("01A", 1000, 6.6),
                        ("01B", 2000, 5.0),
                    ]
                ],
                id="Workflow 2b: SAME measure, DIFFERENT section, cost and beta",
            ),
        ],
    )
    @pytest.mark.fixture_database(
        _custom_measures_test_dir.joinpath("without_custom_measures.db")
    )
    def test_add_custom_measures(
        self,
        custom_measure_dict_list: list[dict],
        custom_measures_vrtool_config: VrtoolConfig,
    ):
        """
        Integration test to verify adding new entries to the `orm.CustomMeasureDetail`
        and related tables under different workflows.
        """
        # 1. Define initial expectations.
        _known_computation_periods = [0, 19, 20, 25, 50, 75, 100]
        _custom_measures_grouped = list(
            (key, list(group))
            for key, group in itertools.groupby(
                custom_measure_dict_list,
                key=itemgetter("MEASURE_NAME", "COMBINABLE_TYPE", "SECTION_NAME"),
            )
        )

        _expected_total_measures = len(
            set(_cm[0][0] + _cm[0][1] for _cm in _custom_measures_grouped)
        )
        _expected_total_custom_measure_details = len(custom_measure_dict_list)
        with open_database(custom_measures_vrtool_config.input_database_path) as _db:
            orm.MeasureResult.delete().execute(_db)
            orm.MeasureResultMechanism.delete().execute(_db)
            orm.MeasureResultSection.delete().execute(_db)
            assert any(orm.MeasureResult.select()) is False
            assert any(orm.MeasureResultMechanism.select()) is False
            assert any(orm.MeasureResultSection.select()) is False
            _expected_total_measures += len(orm.Measure.select())
            _expected_total_custom_measure_details += len(
                orm.CustomMeasureDetail.select()
            )

        # 2. Run test
        _added_measures = add_custom_measures(
            custom_measures_vrtool_config, custom_measure_dict_list
        )

        # 3. Verify final expectations
        assert len(_added_measures) == len(custom_measure_dict_list)

        with open_database(custom_measures_vrtool_config.input_database_path) as _db:
            # Verify the expected amount of `orm.Measure` and `orm.CustomMeasureDetail`
            # entries have been created.
            assert len(orm.Measure.select()) == _expected_total_measures
            assert (
                len(orm.CustomMeasureDetail.select())
                == _expected_total_custom_measure_details
            )

            for _keys_group, _cm_list in _custom_measures_grouped:
                # There should only be one `MeasureResult` for each `CustomMeasureDetail`
                _fm_result = (
                    orm.MeasureResult.select()
                    .join_from(orm.MeasureResult, orm.MeasurePerSection)
                    .join_from(orm.MeasurePerSection, orm.SectionData)
                    .join_from(orm.MeasurePerSection, orm.Measure)
                    .join_from(orm.Measure, orm.CombinableType)
                    .where(
                        (orm.Measure.name == _keys_group[0])
                        & (fn.Upper(orm.CombinableType.name) == _keys_group[1])
                        & (fn.Upper(orm.SectionData.section_name) == _keys_group[2])
                    )
                ).get()
                assert isinstance(_fm_result, orm.MeasureResult)

                # Verify `MeasureResultSection` entries,
                # one per different provided `TIME`.
                assert len(_fm_result.measure_result_section) == len(
                    _known_computation_periods
                )
                for _fm_result_section in _fm_result.measure_result_section:
                    # Costs are the same for a given measure.
                    assert _fm_result_section.cost == _cm_list[0]["COST"]
                    assert _fm_result_section.beta > 0

                # Verify `MeasureResultMechanism` entries,
                # one per different provided `TIME` and mechanisms in `MechanismPerSection`.
                _total_mechs = len(
                    orm.MechanismPerSection.select()
                    .join_from(orm.MechanismPerSection, orm.SectionData)
                    .where(fn.Upper(orm.SectionData.section_name) == _keys_group[2])
                )
                assert len(_fm_result.measure_result_mechanisms) == _total_mechs * len(
                    _known_computation_periods
                )

                # Define expected values
                _expected_mechanism_values = dict()
                for _mechanism_name, _mechanism_dicts in itertools.groupby(
                    _cm_list, key=itemgetter("MECHANISM_NAME")
                ):
                    _mechanism_dicts_list = list(_mechanism_dicts)
                    _time_beta_tuples = [
                        (_cm["TIME"], _cm["BETA"]) for _cm in _mechanism_dicts_list
                    ]
                    _expected_mechanism_values[
                        _mechanism_name
                    ] = CustomMeasureTimeBetaCalculator.get_interpolated_time_beta_collection(
                        _time_beta_tuples, _known_computation_periods
                    )

                for _fm_result_mechanism in _fm_result.measure_result_mechanisms:
                    _mechanism_name = (
                        _fm_result_mechanism.mechanism_per_section.mechanism_name
                    )
                    if _mechanism_name in _expected_mechanism_values:
                        assert (
                            _fm_result_mechanism.beta
                            == _expected_mechanism_values[_mechanism_name][
                                _fm_result_mechanism.time
                            ]
                        )
                    else:
                        _assessment = _fm_result_mechanism.mechanism_per_section.assessment_mechanism_results.where(
                            AssessmentMechanismResult.time == _fm_result_mechanism.time
                        ).get()
                        assert _fm_result_mechanism.beta == pytest.approx(
                            _assessment.beta, rel=1e-6
                        )

    @pytest.mark.fixture_database(_custom_measures_test_dir.joinpath("vrtool_input.db"))
    def test_only_one_measure_added_when_same_measure_per_section(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        def get_current_measure_result_ids_in_db() -> list[int]:
            return list(sorted([_mr.get_id() for _mr in orm.MeasureResult.select()]))

        # 1. Define test data.
        _measure_name = "ROCKS"
        _section_name = "01A"
        _custom_measures = []
        _existing_results_per_measure_ids = []

        with open_database(custom_measures_vrtool_config.input_database_path) as _db:
            _rocks_measure = orm.Measure.get_or_none(orm.Measure.name == _measure_name)
            assert isinstance(_rocks_measure, orm.Measure)
            _measure_per_section = next(
                (
                    _sm
                    for _sm in _rocks_measure.sections_per_measure
                    if _sm.section.section_name == _section_name
                ),
                None,
            )
            assert isinstance(_measure_per_section, orm.MeasurePerSection)
            _existing_results_per_measure_ids = get_current_measure_result_ids_in_db()

            # Add custom measure
            _custom_measures.append(
                {
                    "MEASURE_NAME": _rocks_measure.name,
                    "SECTION_NAME": _measure_per_section.section.section_name,
                    "COMBINABLE_TYPE": _rocks_measure.combinable_type.name.upper(),
                    "MECHANISM_NAME": MechanismEnum.PIPING.name,
                    "COST": 23,
                    "TIME": 0,
                    "BETA": 4.2,
                }
            )

        # 2. Run test.
        _added_custom_measures = add_custom_measures(
            custom_measures_vrtool_config, _custom_measures
        )

        # 3. Verify expectations.
        assert not any(_added_custom_measures)

        with open_database(custom_measures_vrtool_config.input_database_path):
            assert (
                _existing_results_per_measure_ids
                == get_current_measure_result_ids_in_db()
            )

    @pytest.mark.slow
    @pytest.mark.fixture_database(_custom_measures_test_dir.joinpath("vrtool_input.db"))
    def test_import_result_measures_with_custom_measures(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        """
        This test is based on the exported database from
        `test_add_custom_measures[MVP test]`.
        In this test we ONLY focus on verifying whether the `CustomMeasureDetail` and its
        `MeasureResults` are correctly imported.
        """
        # 1. Define test data.
        _measures_section_id = "01A"
        _custom_measure_cost = 50.0

        # Controled values, we use a fix database for this test.
        # These are the id's for the meausre results for the existing
        # `CustomMeasureDetail` entries.
        _custom_measure_detail_ids = [(1, 0)]

        # 2. Run test.
        _imported_data = import_results_measures_for_optimization(
            custom_measures_vrtool_config, _custom_measure_detail_ids
        )

        # 3. Verify expectations.
        assert isinstance(_imported_data, list)
        assert any(_imported_data)
        _imported_with_custom_measures = next(
            _id for _id in _imported_data if _id.section_name == _measures_section_id
        )
        assert isinstance(_imported_with_custom_measures, SectionAsInput)
        assert _imported_with_custom_measures.section_name == _measures_section_id

        _meas_ids = list(
            set(
                (x.measure_result_id, x.year)
                for x in _imported_with_custom_measures.measures
            )
        )
        assert _meas_ids == _custom_measure_detail_ids

        assert len(_imported_with_custom_measures.measures) == 2

        _years = custom_measures_vrtool_config.T
        _expected_betas = np.linspace(7, 4, num=7)

        # Verify each imported measure
        for _measure in _imported_with_custom_measures.measures:
            assert _measure.measure_type == MeasureTypeEnum.CUSTOM
            assert _measure.combine_type == CombinableTypeEnum.FULL
            assert _measure.cost == _custom_measure_cost
            assert _measure.discount_rate == 0.03
            assert _measure.year == 0
            assert _measure.measure_result_id == 1
            if isinstance(_measure, ShSgMeasure):
                assert _measure.base_cost == 0
            else:
                assert _measure.base_cost == _custom_measure_cost

        # Verify betas for `sg_measure` as `MechanismEnum.PIPING` is only
        # compatible for `sg_measures`
        for _sg_measure in _imported_with_custom_measures.sg_measures:
            _overflow_betas = _sg_measure.mechanism_year_collection.get_betas(
                MechanismEnum.PIPING, _years
            )
            assert _overflow_betas == pytest.approx(_expected_betas)

    @pytest.mark.slow
    @pytest.mark.fixture_database(_custom_measures_test_dir.joinpath("vrtool_input.db"))
    def test_run_optimization_with_custom_measures(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        """
        This test is based on the exported database from
        `test_add_custom_measures[MVP test]`.
        We set therefore the known measure results as `_selected_measure_year`.

        For now we only focus on making sure the optimization RUNS.
        """

        # 1. Define test data.
        _optimization_name = "OptimizationWithCustomMeasures"
        _selected_measure_year = [(1, 0)]

        with open_database(custom_measures_vrtool_config.input_database_path) as _db:
            orm.OptimizationRun.delete().execute(_db)
            assert any(orm.OptimizationRun.select()) is False
            assert any(orm.OptimizationSelectedMeasure.select()) is False
            assert any(orm.OptimizationStep.select()) is False
            assert any(orm.OptimizationStepResultMechanism.select()) is False
            assert any(orm.OptimizationStepResultSection.select()) is False

        # 2. Run test.
        _measure_to_run_dict = create_optimization_run_for_selected_measures(
            custom_measures_vrtool_config,
            _optimization_name,
            _selected_measure_year,
        )

        # 3. Verify expectations.
        _expected_runs = len(custom_measures_vrtool_config.design_methods)
        assert len(_measure_to_run_dict) == _expected_runs

        with open_database(custom_measures_vrtool_config.input_database_path):
            assert len(orm.OptimizationRun.select()) == _expected_runs
            for _opt_run in orm.OptimizationRun.select():
                assert _opt_run.discount_rate == 0.03
                assert len(_opt_run.optimization_run_measure_results) == 1
                _selected_measure: orm.OptimizationSelectedMeasure = (
                    _opt_run.optimization_run_measure_results.get()
                )
                assert (
                    _selected_measure.measure_result.measure_per_section.measure.measure_type.name
                    == MeasureTypeEnum.CUSTOM.legacy_name
                )

    @pytest.mark.fixture_database(
        _custom_measures_test_dir.joinpath("without_custom_measures.db")
    )
    def test_add_custom_measures_without_t0_from_csv_raises(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _csv_data = self._custom_measures_test_dir.joinpath(
            "custom_measures_without_t0.csv"
        )
        _csv_list_of_dict = pd.read_csv(_csv_data, delimiter=";").to_dict("records")
        _expected_error = "Missing t0 beta value for Custom Measure"

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            add_custom_measures(custom_measures_vrtool_config, _csv_list_of_dict)

        # 3. Verify expectations.
        assert _expected_error in str(exc_err.value)

    @pytest.mark.fixture_database(_custom_measures_test_dir.joinpath("vrtool_input.db"))
    def test_safe_clear_custom_measure(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        def get_rocks_custom_measure_details() -> list[orm.CustomMeasureDetail]:
            return orm.CustomMeasureDetail.select().where(
                orm.CustomMeasureDetail.measure.name
                == _custom_measure_that_remains_name
            )

        def get_custom_measure_result_ids() -> list[int]:
            return list(
                _mr.get_id()
                for _mr in orm.MeasureResult.select()
                if _mr.measure_type == MeasureTypeEnum.CUSTOM
            )

        def get_existing_optimization_custom_measure(
            existing_measure_results: list[int],
        ) -> bool:
            return any(
                _osm.measure_result_id in existing_measure_results
                for _osm in orm.OptimizationSelectedMeasure.select()
            )

        def check_measure_existence(measure_name: str) -> bool:
            return orm.Measure.get_or_none(orm.Measure.name == measure_name) is not None

        def add_measure_to_database(
            measure_name: str, section_name: str, measure_type: MeasureTypeEnum
        ) -> int:
            _full_combinable = orm.CombinableType.get(
                orm.CombinableType.name == CombinableTypeEnum.FULL.legacy_name
            )
            _measure_that_remains, _created = orm.Measure.get_or_create(
                name=measure_name,
                combinable_type=_full_combinable,
                measure_type=orm.MeasureType.get(
                    orm.MeasureType.name == measure_type.legacy_name
                ),
            )
            _created_measure_x_section = orm.MeasurePerSection.create(
                measure=_measure_that_remains,
                section=orm.SectionData.get(
                    orm.SectionData.section_name == section_name
                ),
            )
            _selected_mechanism_x_section = (
                _created_measure_x_section.section.mechanisms_per_section[0]
            )

            if measure_type == MeasureTypeEnum.CUSTOM:
                for _idx in range(0, 2):
                    _custom_measure_detail = orm.CustomMeasureDetail.create(
                        measure=_measure_that_remains,
                        mechanism_per_section=_selected_mechanism_x_section,
                        time=_idx,
                        cost=42,
                        beta=4.2,
                    )

            _created_measure_result = orm.MeasureResult.create(
                measure_per_section=_created_measure_x_section
            )

            orm.MeasureResultSection.create(
                measure_result=_created_measure_result, beta=42, time=13, cost=24.42
            )
            orm.MeasureResultMechanism.create(
                measure_result=_created_measure_result,
                mechanism_per_section=_selected_mechanism_x_section,
                time=13,
                beta=24.42,
            )
            return _created_measure_result.get_id()

        # 1. Define test data.
        _standard_measure_that_remains_name = "NotACustomMeasure"
        _custom_measure_that_doesnt_remain_name = "BANANAS"
        _custom_measure_that_remains_name = "ROCKS"
        _original_rocks_custom_measure_details_length = 7
        _custom_measure_result_ids = []
        with open_database(custom_measures_vrtool_config.input_database_path):
            _custom_measure_result_ids = get_custom_measure_result_ids()
            assert (
                get_rocks_custom_measure_details()
                == _original_rocks_custom_measure_details_length
            )
            assert any(_custom_measure_result_ids)
            assert get_existing_optimization_custom_measure(_custom_measure_result_ids)
            # Create additional measure results for standard and custom measures.
            _ = add_measure_to_database(
                _standard_measure_that_remains_name,
                "01A",
                MeasureTypeEnum.SOIL_REINFORCEMENT,
            )
            _custom_measure_result_ids.extend(
                [
                    add_measure_to_database(
                        _custom_measure_that_remains_name,
                        "01B",
                        MeasureTypeEnum.CUSTOM,
                    ),
                    add_measure_to_database(
                        _custom_measure_that_doesnt_remain_name,
                        "01A",
                        MeasureTypeEnum.CUSTOM,
                    ),
                ]
            )
            assert len(get_custom_measure_result_ids()) == len(
                _custom_measure_result_ids
            )
            assert (
                get_rocks_custom_measure_details()
                == _original_rocks_custom_measure_details_length + 2
            )

        # 2. Run test.
        safe_clear_custom_measure(custom_measures_vrtool_config)

        # 3. Verify expectations.
        with open_database(custom_measures_vrtool_config.input_database_path):
            _ids = get_custom_measure_result_ids()
            assert len(_ids) == 1
            assert _custom_measure_result_ids[0] == _ids[0]
            assert get_existing_optimization_custom_measure(_ids)

            assert check_measure_existence(_standard_measure_that_remains_name)
            assert check_measure_existence(_custom_measure_that_remains_name)
            assert (
                get_rocks_custom_measure_details()
                == _original_rocks_custom_measure_details_length
            )
            assert (
                check_measure_existence(_custom_measure_that_doesnt_remain_name)
                is False
            )

    @pytest.mark.fixture_database(_custom_measures_test_dir.joinpath("vrtool_input.db"))
    def test_brute_clear_custom_measure(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        def get_custom_measure_result_ids() -> list[int]:
            return list(
                _mr.get_id()
                for _mr in orm.MeasureResult.select()
                if _mr.measure_type == MeasureTypeEnum.CUSTOM
            )

        def get_existing_optimization_custom_measure(
            existing_measure_results: list[int],
        ) -> bool:
            return any(
                _osm.measure_result_id in existing_measure_results
                for _osm in orm.OptimizationSelectedMeasure.select()
            )

        # 1. Define test data.
        _section_name = "01A"
        _measure_that_remains_name = "NotACustomMeasure"
        _measure_result_ids = []
        with open_database(custom_measures_vrtool_config.input_database_path):
            _measure_result_ids = get_custom_measure_result_ids()
            assert any(_measure_result_ids)
            assert get_existing_optimization_custom_measure(_measure_result_ids)
            # Create additional measure results to standard measures.
            assert (
                orm.Measure.get_or_none(orm.Measure.name == _measure_that_remains_name)
                is None
            )
            _measure_that_remains = orm.Measure.create(
                name=_measure_that_remains_name,
                combinable_type=orm.CombinableType.get(),
                measure_type=orm.MeasureType.get(
                    fn.upper(orm.MeasureType.name) != MeasureTypeEnum.CUSTOM.name
                ),
            )
            _created_measure_x_section = orm.MeasurePerSection.create(
                measure=_measure_that_remains,
                section=orm.SectionData.get(
                    orm.SectionData.section_name == _section_name
                ),
            )
            _created_measure_result = orm.MeasureResult.create(
                measure_per_section=_created_measure_x_section
            )
            orm.MeasureResultSection.create(
                measure_result=_created_measure_result, beta=42, time=13, cost=24.42
            )
            orm.MeasureResultMechanism.create(
                measure_result=_created_measure_result,
                mechanism_per_section=_created_measure_x_section.section.mechanisms_per_section[
                    0
                ],
                time=13,
                beta=24.42,
            )

        # 2. Run test.
        brute_clear_custom_measure(custom_measures_vrtool_config)

        # 3. Verify expectations.
        with open_database(custom_measures_vrtool_config.input_database_path):
            assert any(get_custom_measure_result_ids()) is False
            assert any(_measure_result_ids), "The values have been deleted."
            assert (
                get_existing_optimization_custom_measure(_measure_result_ids) is False
            )

            _measure_that_remains = orm.Measure.get_or_none(
                orm.Measure.name == _measure_that_remains_name
            )
            assert isinstance(_measure_that_remains, orm.Measure)
            assert len(_measure_that_remains.sections_per_measure) == 1

            _measure_x_section = _measure_that_remains.sections_per_measure[0]
            assert len(_measure_x_section.measure_per_section_result) == 1

            _measure_result = _measure_x_section.measure_per_section_result[0]
            assert len(_measure_result.measure_result_section) == 1
            assert len(_measure_result.measure_result_mechanisms) == 1

    @pytest.mark.slow
    @pytest.mark.fixture_database(
        _custom_measures_test_dir.joinpath("with_aggregated_measures.db")
    )
    def test_with_aggregated_measures_exports_optimization(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _optimization_name = "OptimizationWithAggregatedCustomMeasures"
        custom_measures_vrtool_config.traject = "16-1"
        custom_measures_vrtool_config.excluded_mechanisms = [
            MechanismEnum.REVETMENT,
            MechanismEnum.HYDRAULIC_STRUCTURES,
        ]
        custom_measures_vrtool_config.externals = test_externals.joinpath(
            "DStabilityConsole"
        )

        with open_database(custom_measures_vrtool_config.input_database_path) as _db:
            orm.OptimizationRun.delete().execute(_db)
            assert any(orm.OptimizationRun.select()) is False
            assert any(orm.OptimizationSelectedMeasure.select()) is False
            assert any(orm.OptimizationStep.select()) is False
            assert any(orm.OptimizationStepResultMechanism.select()) is False
            assert any(orm.OptimizationStepResultSection.select()) is False

        # 2. Run test.
        _all_measure_results = get_all_measure_results_with_supported_investment_years(
            custom_measures_vrtool_config
        )
        _results = ApiRunWorkflows(custom_measures_vrtool_config).run_optimization(
            _optimization_name, _all_measure_results
        )

        # 3. Verify expectations.
        assert isinstance(_results, ResultsOptimization)
        assert len(_results.results_strategies) == 2

        # Greedy strategy
        _greedy_result = next(
            _rs
            for _rs in _results.results_strategies
            if isinstance(_rs, GreedyStrategy)
        )
        assert _greedy_result.OI_horizon == 50
        assert _greedy_result.LCCOption.size > 0
        assert any(_greedy_result.optimization_steps)

        # Target Reliability strategy
        _target_reliability_result = next(
            _rs
            for _rs in _results.results_strategies
            if isinstance(_rs, TargetReliabilityStrategy)
        )
        assert _target_reliability_result.OI_horizon == 50
        assert any(_target_reliability_result.optimization_steps)
