import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from peewee import SqliteDatabase

import vrtool.orm.models as orm_models
from tests import get_test_results_dir, test_data, test_externals, test_results
from vrtool.api import (
    ApiRunWorkflows,
    get_valid_vrtool_config,
    run_full,
    run_step_assessment,
    run_step_measures,
    run_step_optimization,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.measures.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    export_results_measures,
    export_results_optimization,
    open_database,
    vrtool_db,
)
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)


class TestApi:
    def test_given_directory_without_json_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()

        # 2. Run test.
        with pytest.raises(FileNotFoundError) as exception_error:
            get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "No json config file found in the model directory {}.".format(_input_dir)

    def test_given_directory_with_too_many_jsons_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if _input_dir.exists():
            shutil.rmtree(_input_dir)

        _input_dir.mkdir(parents=True)
        Path.joinpath(_input_dir, "first.json").touch()
        Path.joinpath(_input_dir, "second.json").touch()

        # 2. Run test.
        with pytest.raises(ValueError) as exception_error:
            get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "More than one json file found in the directory {}. Only one json at the root directory supported.".format(
            _input_dir
        )

    def test_given_directory_with_valid_config_returns_vrtool_config(self):
        # 1. Define test data.
        _input_dir = test_data / "vrtool_config"
        assert _input_dir.exists()

        # 2. Run test.
        _vrtool_config = get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert isinstance(_vrtool_config, VrtoolConfig)
        assert _vrtool_config.traject == "MyCustomTraject"
        assert _vrtool_config.input_directory == _input_dir
        assert _vrtool_config.output_directory == _input_dir / "results"


class TestApiRunWorkflows:
    @pytest.mark.parametrize(
        "vrtool_config",
        [
            pytest.param(None, id="No provided VrtoolConfig"),
            pytest.param(VrtoolConfig("just_a_db.db"), id="With dummy VrtoolConfig"),
        ],
    )
    def test_init_with_different_vrtool_config(self, vrtool_config: VrtoolConfig):
        _api_run_workflows = ApiRunWorkflows(vrtool_config)
        assert isinstance(_api_run_workflows, ApiRunWorkflows)


# Defining acceptance test cases so they are accessible from the `TestAcceptance` class.

_acceptance_all_steps_test_cases = [
    pytest.param(
        ("TestCase1_38-1_no_housing", "38-1", ["Revetment", "HydraulicStructures"]),
        id="Traject 38-1, no housing",
    ),
    pytest.param(
        (
            "TestCase1_38-1_no_housing_stix",
            "38-1",
            ["Revetment", "HydraulicStructures"],
        ),
        id="Traject 38-1, no housing, with dstability",
    ),
    pytest.param(
        (
            "TestCase2_38-1_overflow_no_housing",
            "38-1",
            ["Revetment", "HydraulicStructures"],
        ),
        id="Traject 38-1, no-housing, with overflow",
    ),
    pytest.param(
        ("TestCase1_38-1_revetment", "38-1", ["HydraulicStructures"]),
        id="Traject 38-1, with revetment, case 1",
    ),
    pytest.param(
        ("TestCase3_38-1_revetment", "38-1", ["HydraulicStructures"]),
        id="Traject 38-1, with revetment, including bundling",
    ),
    pytest.param(
        ("TestCase4_38-1_revetment_small", "38-1", ["HydraulicStructures"]),
        id="Traject 38-1, two sections with revetment",
    ),
]


@pytest.mark.slow
class TestApiRunWorkflowsAcceptance:
    vrtool_db_default_name = "vrtool_input.db"

    @pytest.fixture
    def valid_vrtool_config(self, request: pytest.FixtureRequest) -> VrtoolConfig:
        _casename, _traject, _excluded_mechanisms = request.param
        _test_input_directory = Path.joinpath(test_data, _casename)
        assert _test_input_directory.exists()

        _test_results_directory = get_test_results_dir(request).joinpath(_casename)
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)
        _test_results_directory.mkdir(parents=True)

        # Define the VrtoolConfig
        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _traject
        _test_config.excluded_mechanisms = _excluded_mechanisms
        _test_config.externals = test_externals

        # We need to create a copy of the database on the input directory.
        _test_db_name = "test_{}.db".format(
            hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
        )
        _test_config.input_database_name = _test_db_name

        # Create a copy of the database to avoid parallelization runs locked databases.
        _reference_db_file = _test_input_directory.joinpath(self.vrtool_db_default_name)
        assert _reference_db_file.exists(), "No database found at {}.".format(
            _reference_db_file
        )

        if _test_config.input_database_path.exists():
            # Somehow it was not removed in the previous test run.
            _test_config.input_database_path.unlink(missing_ok=True)

        shutil.copy(_reference_db_file, _test_config.input_database_path)
        assert (
            _test_config.input_database_path.exists()
        ), "No database found at {}.".format(_reference_db_file)

        yield _test_config

        # Make sure that the database connection will be closed even if the test fails.
        if isinstance(vrtool_db, SqliteDatabase) and not vrtool_db.is_closed():
            vrtool_db.close()

        # Copy the test database to the results directory so it can be manually reviewed.
        if _test_config.input_database_path.exists():
            _results_db_name = _test_config.output_directory.joinpath(
                "vrtool_result.db"
            )
            shutil.move(_test_config.input_database_path, _results_db_name)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=True,
    )
    def test_run_step_assessment_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        clear_assessment_results(valid_vrtool_config)
        _validator = RunStepAssessmentValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_assessment(valid_vrtool_config)

        # 3. Verify expectations.
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # 4. Validate exporting results is possible
        _validator.validate_safety_assessment_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=True,
    )
    def test_run_step_measures_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _validator = RunStepMeasuresValidator()

        clear_measure_results(valid_vrtool_config)
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_measures(valid_vrtool_config)

        # 3. Verify expectations.
        _validator.validate_measure_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases
        + [
            pytest.param(
                ("TestCase3_38-1_small", "38-1", ["Revetment", "HydraulicStructures"]),
                id="Traject 38-1, two sections",
                marks=[pytest.mark.skip(reason="Missing input database.")],
            )
        ],
        indirect=True,
    )
    def test_run_step_optimization_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # TODO: Extend this test with ALL the `_acceptance_all_steps_test_cases`
        # once `test_run_step_measures_given_valid_vrtool_config` works correctly.
        # 1. Define test data.
        # We reuse existing measure results, but we clear the optimization ones.
        clear_optimization_results(valid_vrtool_config)

        _validator = RunStepOptimizationValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        _measures_results = _validator.get_test_measure_result_ids(valid_vrtool_config)

        # 2. Run test.
        run_step_optimization(valid_vrtool_config, _measures_results)

        # 3. Verify expectations.
        _validator.validate_optimization_results(valid_vrtool_config)
        RunFullValidator().validate_acceptance_result_cases(
            valid_vrtool_config.output_directory,
            valid_vrtool_config.input_directory.joinpath("reference"),
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        [
            pytest.param(
                ("TestCase3_38-1_small", "38-1", ["Revetment", "HydraulicStructures"]),
                id="Traject 38-1, two sections",
            ),
        ],
        indirect=True,
    )
    def test_run_optimization_old_approach(self, valid_vrtool_config: VrtoolConfig):
        # TODO: Get the input database of `TestCase3_38-1_small` and run
        # the test in `test_run_step_optimization_given_valid_vrtool_config` instead.
        _test_reference_path = valid_vrtool_config.input_directory / "reference"

        _shelve_path = valid_vrtool_config.input_directory / "shelves"
        _results_assessment = ResultsSafetyAssessment()
        _results_assessment.load_results(
            alternative_path=_shelve_path / "AfterStep1.out"
        )
        _results_measures = ResultsMeasures()

        _results_measures.vr_config = valid_vrtool_config
        _results_measures.selected_traject = _results_assessment.selected_traject

        _results_measures.load_results(alternative_path=_shelve_path / "AfterStep2.out")
        _results_optimization = RunOptimization(_results_measures).run()

        export_results_measures(_results_measures)
        _results_optimization.vr_config = valid_vrtool_config
        export_results_optimization(_results_optimization)

        RunFullValidator().validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=True,
    )
    @pytest.mark.skip(reason="Phased out in favor of running each step individually.")
    def test_run_full_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_reference_path = valid_vrtool_config.input_directory / "reference"
        assert _test_reference_path.exists()

        # 2. Run test.
        run_full(valid_vrtool_config)

        # 3. Verify final expectations.
        RunFullValidator().validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )


class RunStepAssessmentValidator:
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(orm_models.AssessmentMechanismResult.select())
        assert not any(orm_models.AssessmentSectionResult.select())
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_safety_assessment_results(self, valid_vrtool_config: VrtoolConfig):
        # Get database paths.
        _reference_database_path = valid_vrtool_config.input_database_path.with_name(
            TestApiRunWorkflowsAcceptance.vrtool_db_default_name
        )
        assert (
            _reference_database_path != valid_vrtool_config.input_database_path
        ), "Reference and result database point to the same Path {}.".path(
            valid_vrtool_config.input_database_path
        )

        def load_assessment_reliabilities(vrtool_db: Path) -> dict[str, pd.DataFrame]:
            _connected_db = open_database(vrtool_db)
            _assessment_reliabilities = dict(
                (_sd, DikeSectionImporter.import_assessment_reliability_df(_sd))
                for _sd in orm_models.SectionData.select()
                .join(orm_models.DikeTrajectInfo)
                .where(
                    orm_models.SectionData.dike_traject.traject_name
                    == valid_vrtool_config.traject
                )
            )
            _connected_db.close()
            return _assessment_reliabilities

        _result_assessment = load_assessment_reliabilities(
            valid_vrtool_config.input_database_path
        )
        _reference_assessment = load_assessment_reliabilities(_reference_database_path)

        assert any(
            _reference_assessment.items()
        ), "No reference assessments were loaded."
        _errors = []
        for _ref_key, _ref_dataframe in _reference_assessment.items():
            _res_dataframe = _result_assessment.get(_ref_key, pd.DataFrame())
            if _res_dataframe.empty:
                _errors.append(
                    "Section {} has no reliability results.".format(_ref_key)
                )
                continue
            pd.testing.assert_frame_equal(_ref_dataframe, _res_dataframe)
        if _errors:
            pytest.fail("\n".join(_errors))


class RunStepMeasuresValidator:
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(orm_models.MeasureResult.select())
        assert not any(orm_models.MeasureResultParameter.select())

        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_measure_results(self, valid_vrtool_config: VrtoolConfig):
        """
        {
            "section_id":
                "measure_id":
                    "frozenset[measure_result_with_params]": reliability
        }
        """

        # Get database paths.
        _reference_database_path = valid_vrtool_config.input_database_path.with_name(
            TestApiRunWorkflowsAcceptance.vrtool_db_default_name
        )
        assert (
            _reference_database_path != valid_vrtool_config.input_database_path
        ), "Reference and result database point to the same Path {}.".path(
            valid_vrtool_config.input_database_path
        )

        def load_measures_reliabilities(
            vrtool_db: Path,
        ) -> dict[str, dict[tuple, pd.DataFrame]]:
            _connected_db = open_database(vrtool_db)
            _m_reliabilities = defaultdict(dict)
            for _measure_result in orm_models.MeasureResult.select():
                _mxs = _measure_result.measure_per_section
                _reliability_df = MeasureResultImporter.import_measure_reliability_df(
                    _measure_result
                )
                _available_parameters = frozenset(
                    (mrp.name, mrp.value)
                    for mrp in _measure_result.measure_result_parameters
                )
                if (
                    _available_parameters
                    in _m_reliabilities[
                        (_mxs.measure.name, _mxs.section.section_name)
                    ].keys()
                ):
                    _keys_values = [
                        f"{k}={v}" for k, v in _available_parameters.items()
                    ]
                    _as_string = ", ".join(_keys_values)
                    pytest.fail(
                        "Measure reliability contains twice the same parameters {}.".format(
                            _as_string
                        )
                    )
                _m_reliabilities[(_mxs.measure.name, _mxs.section.section_name)][
                    _available_parameters
                ] = _reliability_df
            _connected_db.close()
            return _m_reliabilities

        _result_assessment = load_measures_reliabilities(
            valid_vrtool_config.input_database_path
        )
        _reference_assessment = load_measures_reliabilities(_reference_database_path)

        assert any(
            _reference_assessment.items()
        ), "No reference assessments were loaded."
        _errors = []
        for _ref_key, _ref_section_measure_dict in _reference_assessment.items():
            # Iterate over each dictiory entry,
            # which represents ALL the measure results (the values)
            # of a given `MeasurePerSection` (the key).
            _res_section_measure_dict = _result_assessment.get(_ref_key, dict())
            if not any(_res_section_measure_dict.items()):
                _errors.append(
                    "Measure {} = Section {}, have no reliability results.".format(
                        _ref_key[0], _ref_key[1]
                    )
                )
                continue
            for (
                _ref_params,
                _ref_measure_result_reliability,
            ) in _ref_section_measure_dict.items():
                # Iterate over each dictionary entry,
                # which represents the measure reliability results (the values as `pd.DataFrame`)
                # for a given set of parameters represented as `dict` (the keys)
                _res_measure_result_reliability = _res_section_measure_dict.get(
                    _ref_params, pd.DataFrame()
                )
                if _res_measure_result_reliability.empty:
                    _parameters = [f"{k}={v}" for k, v in _ref_params]
                    _parameters_as_str = ", ".join(_parameters)
                    _errors.append(
                        "Measure {} = Section {}, Parameters: {}, have no reliability results".format(
                            _ref_key[0], _ref_key[1], _parameters_as_str
                        )
                    )
                    continue
                pd.testing.assert_frame_equal(
                    _ref_measure_result_reliability, _res_measure_result_reliability
                )
        if _errors:
            pytest.fail("\n".join(_errors))


class RunStepOptimizationValidator:
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        assert any(orm_models.MeasureResult.select())
        assert not any(orm_models.OptimizationRun)
        assert not any(orm_models.OptimizationSelectedMeasure)
        assert not any(orm_models.OptimizationStep)
        assert not any(orm_models.OptimizationStepResult)

        _connected_db.close()

    def get_test_measure_result_ids(
        self, valid_vrtool_config: VrtoolConfig
    ) -> list[int]:
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        _id_list = [mr.get_id() for mr in orm_models.MeasureResult.select()]
        _connected_db.close()
        return _id_list

    def validate_optimization_results(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        # For now just check that there are outputs.
        assert any(orm_models.OptimizationRun.select())
        assert any(orm_models.OptimizationSelectedMeasure.select())
        assert any(orm_models.OptimizationStep.select())
        assert any(orm_models.OptimizationStepResult.select())
        _connected_db.close()


class RunFullValidator:
    def validate_acceptance_result_cases(
        self, test_results_dir: Path, test_reference_dir: Path
    ):
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]
        comparison_errors = []
        for file in files_to_compare:
            reference = pd.read_csv(
                test_reference_dir.joinpath("results", file), index_col=0
            )
            result = pd.read_csv(test_results_dir / file, index_col=0)
            try:
                assert_frame_equal(reference, result, atol=1e-6, rtol=1e-6)
            except Exception:
                comparison_errors.append("{} is different.".format(file))
        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )
