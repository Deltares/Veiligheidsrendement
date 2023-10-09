import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from peewee import SqliteDatabase, fn

from tests import get_test_results_dir, test_data, test_externals, test_results
from vrtool.api import (
    get_valid_vrtool_config,
    run_full,
    run_step_assessment,
    run_step_measures,
    run_step_optimization,
)
from vrtool.common.enums import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import (
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


# Defining acceptance test cases so they are accessible from the `TestAcceptance` class.

_acceptance_all_steps_test_cases = [
    pytest.param(
        ("TestCase1_38-1_no_housing", "38-1", ["REVETMENT", "HYDRAULIC_STRUCTURES"]),
        id="Traject 38-1, no housing",
    ),
    pytest.param(
        (
            "TestCase1_38-1_no_housing_stix",
            "38-1",
            ["REVETMENT", "HYDRAULIC_STRUCTURES"],
        ),
        id="Traject 38-1, no housing, with dstability",
    ),
    pytest.param(
        (
            "TestCase2_38-1_overflow_no_housing",
            "38-1",
            ["REVETMENT", "HYDRAULIC_STRUCTURES"],
        ),
        id="Traject 38-1, no-housing, with overflow",
    ),
    pytest.param(
        ("TestCase1_38-1_revetment", "38-1", ["HYDRAULIC_STRUCTURES"]),
        id="Traject 38-1, with revetment, case 1",
    ),
    pytest.param(
        ("TestCase3_38-1_revetment", "38-1", ["HYDRAULIC_STRUCTURES"]),
        id="Traject 38-1, with revetment, including bundling",
    ),
    pytest.param(
        ("TestCase4_38-1_revetment_small", "38-1", ["HYDRAULIC_STRUCTURES"]),
        id="Traject 38-1, two sections with revetment",
    ),
]
_acceptance_optimization_test_cases = [
    _acceptance_all_steps_test_cases[0],
    pytest.param(
        ("TestCase3_38-1_small", "38-1", ["REVETMENT", "HYDRAULIC_STRUCTURES"]),
        id="Traject 38-1, two sections",
    ),
]
_acceptance_measure_test_cases = [
    _acceptance_all_steps_test_cases[0],
    _acceptance_all_steps_test_cases[2],
]


@pytest.mark.slow
class TestRunWorkflows:
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
        _reference_db_file = _test_input_directory.joinpath("vrtool_input.db")
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
        _acceptance_measure_test_cases,
        indirect=True,
    )
    @pytest.mark.skip(
        reason="Extremely slow due to the validation, these tests are validated in 'run_full'."
    )
    def test_run_step_measures_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _validator = RunStepMeasuresValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_measures(valid_vrtool_config)

        # 3. Verify expectations.
        _validator.validate_measure_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_optimization_test_cases,
        indirect=True,
    )
    def test_run_optimization_old_approach(self, valid_vrtool_config: VrtoolConfig):
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
    @pytest.mark.skip(reason="Needs to be implemented by VRTOOL-222.")
    def test_run_step_optimization_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _test_reference_path = valid_vrtool_config.input_directory / "reference"
        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        run_step_optimization(valid_vrtool_config)

        # 3. Verify expectations.
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        RunFullValidator().validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=True,
    )
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
        assert not any(AssessmentMechanismResult.select())
        assert not any(AssessmentSectionResult.select())
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_safety_assessment_results(self, valid_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        _test_reference_path = valid_vrtool_config.input_directory / "reference"
        assert _test_reference_path.exists()

        # 2. Load reference as pandas dataframe.
        _reference_df = pd.read_csv(
            _test_reference_path.joinpath("InitialAssessment_Betas.csv"), header=0
        )

        # 3. Validate each of the rows.
        # Open the database (whose path has been overwritten with the backup).
        # This will overwrite the global variable vrtool_db.
        # In the test's teardown we can ensure closing its connection.
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        self.validate_mechanism_per_section_initial_assessment(
            _reference_df[_reference_df["mechanism"] != "Section"],
            valid_vrtool_config,
        )
        self.validate_section_data_initial_assessment(
            _reference_df[_reference_df["mechanism"] == "Section"],
            valid_vrtool_config,
        )
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_section_data_initial_assessment(
        self, reference_df: pd.DataFrame, vrtool_config: VrtoolConfig
    ):
        assert len(AssessmentSectionResult.select()) == (
            len(reference_df.index) * len(vrtool_config.T)
        )
        for _, row in reference_df.iterrows():
            _section_data = SectionData.get(SectionData.section_name == row["name"])
            for _t_column in vrtool_config.T:
                _assessment_result = AssessmentSectionResult.get_or_none(
                    (AssessmentSectionResult.section_data == _section_data)
                    & (AssessmentSectionResult.time == int(_t_column))
                )
                assert isinstance(
                    _assessment_result, AssessmentSectionResult
                ), "Initial assessment not found for dike section {}, t {}.".format(
                    row["name"], _t_column
                )
                assert _assessment_result.beta == pytest.approx(
                    row[str(_t_column)], 0.00000001
                ), "Mismatched values for section {}, t {}".format(
                    row["name"], _t_column
                )

    def validate_mechanism_per_section_initial_assessment(
        self, reference_df: pd.DataFrame, vrtool_config: VrtoolConfig
    ):
        assert len(AssessmentMechanismResult.select()) == (
            len(reference_df.index) * len(vrtool_config.T)
        )
        for _, row in reference_df.iterrows():
            _section_data = SectionData.get(SectionData.section_name == row["name"])
            _mechanism = MechanismEnum.get_enum(row["mechanism"])
            _mech_inst = Mechanism.get(
                Mechanism.name << [_mechanism.name, _mechanism.get_old_name()]
            )
            _mechanism_x_section = MechanismPerSection.get_or_none(
                (MechanismPerSection.section == _section_data)
                & (MechanismPerSection.mechanism == _mech_inst)
            )
            assert isinstance(_mechanism_x_section, MechanismPerSection)
            for _t_column in vrtool_config.T:
                _assessment_result = AssessmentMechanismResult.get_or_none(
                    (
                        AssessmentMechanismResult.mechanism_per_section
                        == _mechanism_x_section
                    )
                    & (AssessmentMechanismResult.time == int(_t_column))
                )
                assert isinstance(
                    _assessment_result, AssessmentMechanismResult
                ), "No entry found for section {} and mechanism {}".format(
                    row["name"], _mechanism
                )
                assert _assessment_result.beta == pytest.approx(
                    row[str(_t_column)], 0.00000001
                ), "Missmatched values for section {}, mechanism {}, t {}".format(
                    row["name"], _mechanism, _t_column
                )


class RunStepMeasuresValidator:
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_measure_results(self, valid_vrtool_config: VrtoolConfig):
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # 1. Define test data.
        _test_reference_path = (
            valid_vrtool_config.input_directory / "reference" / "results"
        )
        assert _test_reference_path.exists()

        _reference_file_paths = list(
            _test_reference_path.glob("*_Options_Veiligheidsrendement.csv")
        )
        _reference_section_names = [
            self.get_section_name(_file_path) for _file_path in _reference_file_paths
        ]

        # 2. Open the database to retrieve the section names to read the references from
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        # 3. Verify there are no measures whose section's reference file does not exist.
        _sections_with_measures = list(
            set(
                _measure_per_section.section.section_name
                for _measure_per_section in MeasurePerSection.select(
                    MeasurePerSection, SectionData
                ).join(SectionData)
            )
        )
        assert all(_rds in _sections_with_measures for _rds in _reference_section_names)

        # 4. Load reference as pandas dataframe.
        total_nr_of_measure_results = 0
        total_nr_of_measure_result_parameters = 0
        for reference_file_path in _reference_file_paths:
            reference_data = self.get_reference_measure_result_data(reference_file_path)

            section_name = self.get_section_name(reference_file_path)
            section = SectionData.get_or_none(SectionData.section_name == section_name)
            assert section, "SectionData not found for dike section {}.".format(
                section_name
            )
            self.validate_measure_result_per_section(reference_data, section)

            # The total amount of results for a single section must be equal to the amount
            #  of years * the amount of measures that are not of the "class" combined
            nr_of_years = reference_data[("Section",)].shape[1]
            total_nr_of_measure_results += len(reference_data.index) * nr_of_years

            # The total amount of measure parameters are equal to the amount of rows in the reference
            # data where the dcrest and dberm are unequal to -999 * the amount of years * nr of parameters
            # (which is just dcrest and dberm) for the reference data
            total_nr_of_measure_result_parameters += (
                reference_data[
                    (reference_data[("dcrest",)] != -999)
                    & (reference_data[("dberm",)] != -999)
                ].shape[0]
                * nr_of_years
                * 2
            )

        assert len(MeasureResult.select()) == len(reference_data.index)
        assert len(MeasureResultSection.select()) == total_nr_of_measure_results
        assert (
            len(MeasureResultParameter.select())
            == total_nr_of_measure_result_parameters
        )
        if not _connected_db.is_closed():
            _connected_db.close()

    def get_section_name(self, file_path: Path) -> str:
        return file_path.name.split("_")[0]

    def get_reference_measure_result_data(
        self, reference_file_path: Path
    ) -> pd.DataFrame:
        _read_reference_df = pd.read_csv(reference_file_path, header=[0, 1])

        # Filter reference values as we are not interested in the reliabilities for the individual failure mechanisms
        _filtered_reference_df = _read_reference_df.loc[
            :,
            _read_reference_df.columns.get_level_values(0).isin(
                ["ID", "type", "class", "year", "dcrest", "dberm", "cost", "Section"]
            ),
        ]

        # Rename the "unnamed" columns or the reference data cannot be filtered
        normalised_columns = [
            (column[0], "") if column[0] != "Section" else (column[0], column[1])
            for column in _filtered_reference_df.columns.tolist()
        ]
        new_columns = pd.MultiIndex.from_tuples(normalised_columns)
        _filtered_reference_df.columns = new_columns

        # We are also not interested in the combined measure results, because these are derived solutions and are not
        # exported by the measure exporter
        _filtered_reference_df = _filtered_reference_df[
            _filtered_reference_df[("class",)] != "combined"
        ]

        return _filtered_reference_df

    def validate_measure_result_per_section(
        self, reference_df: pd.DataFrame, section: SectionData
    ) -> None:
        measure_result_lookup: dict[tuple(int, float, float), MeasureResult] = {}
        unique_measure_ids = set()
        for _, row in reference_df.iterrows():
            casted_measure_id = int(row[("ID",)])
            if not (casted_measure_id in unique_measure_ids):
                unique_measure_ids.add(casted_measure_id)
                measure_result_lookup.clear()  # Reset the lookup for each measure or the lookup maintains the  entries of the previous measure

                _measure = Measure.get_by_id(casted_measure_id)
                _measure_per_section = MeasurePerSection.get_or_none(
                    (MeasurePerSection.section == section)
                    & (MeasurePerSection.measure == _measure)
                )

            dberm = float(row[("dberm",)].item())
            dcrest = float(row[("dcrest",)].item())

            reference_section_reliabilities = row[("Section",)]
            for year in reference_section_reliabilities.index:
                casted_year = int(year)
                if self.is_soil_reinforcement_measure(dberm, dcrest):
                    self.fill_measure_result_lookup_for_soil_reinforcement_measures(
                        measure_result_lookup, _measure_per_section, casted_year
                    )

                _measure_result = self.get_measure_result(
                    measure_result_lookup,
                    _measure_per_section,
                    casted_year,
                    row["type"],
                    casted_measure_id,
                    dberm,
                    dcrest,
                )
                _measure_result_section = MeasureResultSection.get_or_none(
                    measure_result=_measure_result, time=casted_year
                )
                assert isinstance(_measure_result_section, MeasureResultSection)

                assert _measure_result_section.beta == pytest.approx(
                    reference_section_reliabilities[year], 0.00000001
                ), "Mismatched beta for section {} and measure result {} with id {}, dberm {}, dcrest {}, and year {}".format(
                    section.section_name,
                    row[("type",)],
                    casted_measure_id,
                    dberm,
                    dcrest,
                    casted_year,
                )

                assert _measure_result_section.cost == float(
                    row[("cost",)]
                ), "Mismatched cost for section {} and measure result {} with id {}, dberm {}, dcrest {}, and year {}".format(
                    section.section_name,
                    row[("type",)],
                    casted_measure_id,
                    dberm,
                    dcrest,
                    casted_year,
                )

    def fill_measure_result_lookup_for_soil_reinforcement_measures(
        self,
        lookup: dict[tuple[int, float, float], MeasureResult],
        measure_per_section: MeasurePerSection,
        year: int,
    ) -> None:
        _measure_results = MeasureResult.select().where(
            (MeasureResult.measure_per_section == measure_per_section)
        )

        for _found_result in _measure_results:
            _dberm_parameter = MeasureResultParameter.get(
                (MeasureResultParameter.measure_result == _found_result)
                & (MeasureResultParameter.name == "DBERM")
            )

            _dcrest_parameter = MeasureResultParameter.get(
                (MeasureResultParameter.measure_result == _found_result)
                & (MeasureResultParameter.name == "DCREST")
            )

            _dberm = _dberm_parameter.value
            _dcrest = _dcrest_parameter.value
            if not ((year, _dberm, _dcrest) in lookup):
                lookup[(year, _dberm, _dcrest)] = _found_result

    def get_measure_result(
        self,
        soil_reinforcement_measures_lookup: dict[
            tuple[int, float, float], MeasureResult
        ],
        measure_per_section: MeasurePerSection,
        year: int,
        measure_type: str,
        measure_id: str,
        dberm: float,
        dcrest: float,
    ) -> MeasureResult:
        if self.is_soil_reinforcement_measure(
            dberm, dcrest
        ):  # Filter on parameter to get the right measure result
            _measure_result = soil_reinforcement_measures_lookup.get(
                (year, dberm, dcrest), None
            )

            # Assert that the associated parameters are only DCREST and DBERM
            assert isinstance(
                _measure_result, MeasureResult
            ), "No entry found for section {} and measure result {} with id {} and year {}".format(
                measure_per_section.section.section_name, measure_type, measure_id, year
            )
            assert len(_measure_result.measure_result_parameters.select()) == 2

            return _measure_result

        _measure_result = MeasureResult.get_or_none(
            (MeasureResult.measure_per_section == measure_per_section)
        )

        assert isinstance(
            _measure_result, MeasureResult
        ), "No entry found for section {} and measure result {} with id {} and year {}".format(
            measure_per_section.section.section_name, measure_type, measure_id, year
        )
        assert not any(_measure_result.measure_result_parameters.select())

        return _measure_result

    def is_soil_reinforcement_measure(self, dberm: float, dcrest: float) -> bool:
        return dberm != -999 and dcrest != -999


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
