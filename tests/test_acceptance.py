import shutil
from pathlib import Path
from re import search

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from peewee import SqliteDatabase, fn

from tests import get_test_results_dir, test_data, test_externals
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult, MeasureResultParameter
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    export_results_measures,
    export_results_optimization,
    export_results_safety_assessment,
    get_dike_traject,
    open_database,
    vrtool_db,
)
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel

# Defining acceptance test cases so they are accessible from the `TestAcceptance` class.
_available_mechanisms = ["Overflow", "StabilityInner", "Piping", "Revetment"]

_acceptance_all_steps_test_cases = [
    pytest.param(
        ("TestCase1_38-1_no_housing", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, no housing",
    ),
    pytest.param(
        ("TestCase1_38-1_no_housing_stix", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, no housing, with dstability",
    ),
    pytest.param(
        ("TestCase2_38-1_overflow_no_housing", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, no-housing, with overflow",
    ),
    pytest.param(
        ("TestCase1_38-1_revetment", "38-1", _available_mechanisms),
        id="Traject 38-1, with revetment, case 1",
    ),
    pytest.param(
        ("TestCase3_38-1_revetment", "38-1", _available_mechanisms),
        id="Traject 38-1, with revetment, including bundling",
    ),
    pytest.param(
        ("TestCase4_38-1_revetment_small", "38-1", _available_mechanisms),
        id="Traject 38-1, two sections with revetment",
    ),
]

_acceptance_optimization_test_cases = [
    _acceptance_all_steps_test_cases[0],
    pytest.param(
        ("TestCase3_38-1_small", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, two sections",
    ),
]
_acceptance_measure_test_cases = [
    _acceptance_all_steps_test_cases[0],
    _acceptance_all_steps_test_cases[2],
]


@pytest.mark.slow
class TestAcceptance:
    def _validate_acceptance_result_cases(
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

    @pytest.fixture
    def valid_vrtool_config(self, request: pytest.FixtureRequest) -> VrtoolConfig:
        _casename, _traject, _mechanisms = request.param
        _test_input_directory = Path.joinpath(test_data, _casename)
        assert _test_input_directory.exists()

        _test_results_directory = get_test_results_dir(request) / _casename
        _test_db_name = f"{request.node.name}.db"
        if "[" in request.node.name:
            # It is a parametrized case:
            _node_parts = search(r"(.*)\[(.*)\]", request.node.name)
            _node_case = _node_parts.group(2).strip()
            _test_results_directory = _test_results_directory / _node_case.replace(
                ",", "_"
            ).replace(" ", "_")
            _node_name = _node_parts.group(1).strip()
            _test_db_name = f"{_node_name}.db"
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)

        _test_results_directory.mkdir(parents=True)

        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _traject
        _test_config.mechanisms = _mechanisms
        _test_config.externals = test_externals

        # Create a copy of the database to avoid parallelization runs locked databases.
        _db_file = _test_input_directory.joinpath("vrtool_input.db")
        assert _db_file.exists(), "No database found at {}.".format(_db_file)

        _test_config.input_database_name = _test_db_name
        _tst_db_file = _test_config.input_database_path
        _tst_db_file.unlink(missing_ok=True)
        shutil.copy(_db_file, _tst_db_file)
        assert _tst_db_file.exists(), "No database found at {}.".format(_db_file)

        yield _test_config

        # Copy the test database to the results directory
        if _tst_db_file.exists():
            shutil.move(_tst_db_file, _test_config.output_directory)

        # Make sure that the database connection will be closed even if the test fails.
        if isinstance(vrtool_db, SqliteDatabase) and not vrtool_db.is_closed():
            vrtool_db.close()

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=["valid_vrtool_config"],
    )
    def test_run_full_model(self, valid_vrtool_config: VrtoolConfig):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_reference_path = valid_vrtool_config.input_directory / "reference"
        assert _test_reference_path.exists()

        _test_traject = get_dike_traject(valid_vrtool_config)

        # 2. Run test.
        _optimization_results = RunFullModel(valid_vrtool_config, _test_traject).run()

        # export measures
        _rm = ResultsMeasures()
        _rm.solutions_dict = _optimization_results.results_solutions
        _rm.vr_config = valid_vrtool_config
        export_results_measures(_rm)
        # export optimization

        # 3. Verify final expectations.
        self._validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_all_steps_test_cases,
        indirect=["valid_vrtool_config"],
    )
    def test_run_safety_assessment_and_save_initial_assessment(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _test_traject = get_dike_traject(valid_vrtool_config)
        assert not any(AssessmentMechanismResult.select())
        assert not any(AssessmentSectionResult.select())

        # 2. Run test.
        _results = RunSafetyAssessment(valid_vrtool_config, _test_traject).run()

        # 3. Verify expectations.
        assert isinstance(_results, ResultsSafetyAssessment)
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # NOTE: Ideally this is done with the context manager and a db.savepoint() transaction.
        # However, this is not possible as the connection will be closed during the export_initial_assessment.
        # Causing an error as the transaction requires said connection to be open.
        # Therefore the following has been found as the only possible way to assess whether the results are
        # written in the database without affecting other tests from using this db.
        _bck_db_name = "bck_db.db"
        _bck_db_filepath = valid_vrtool_config.input_database_path.with_name(
            _bck_db_name
        )
        shutil.copyfile(valid_vrtool_config.input_database_path, _bck_db_filepath)
        _results.vr_config.input_directory = valid_vrtool_config.input_directory
        _results.vr_config.input_database_name = _bck_db_name

        # 4. Validate exporting results is possible
        clear_assessment_results(valid_vrtool_config)
        export_results_safety_assessment(_results)
        self.validate_safety_assessment_results(valid_vrtool_config)

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
        open_database(valid_vrtool_config.input_database_path)
        self.validate_mechanism_per_section_initial_assessment(
            _reference_df[_reference_df["mechanism"] != "Section"],
            valid_vrtool_config,
        )
        self.validate_section_data_initial_assessment(
            _reference_df[_reference_df["mechanism"] == "Section"],
            valid_vrtool_config,
        )

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
                ), "Missmatched values for section {}, t {}".format(
                    row["name"], _t_column
                )

    def validate_mechanism_per_section_initial_assessment(
        self, reference_df: pd.DataFrame, vrtool_config: VrtoolConfig
    ):
        assert len(AssessmentMechanismResult.select()) == (
            len(reference_df.index) * len(vrtool_config.T)
        )
        for _, row in reference_df.iterrows():
            _mechanism_name = row["mechanism"]
            _section_data = SectionData.get(SectionData.section_name == row["name"])
            _mechanism = Mechanism.get(
                fn.Upper(Mechanism.name) == _mechanism_name.upper()
            )
            _mechanism_x_section = MechanismPerSection.get_or_none(
                (MechanismPerSection.section == _section_data)
                & (MechanismPerSection.mechanism == _mechanism)
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
                    row["name"], _mechanism_name
                )
                assert _assessment_result.beta == pytest.approx(
                    row[str(_t_column)], 0.00000001
                ), "Missmatched values for section {}, mechanism {}, t {}".format(
                    row["name"], _mechanism_name, _t_column
                )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_optimization_test_cases,
        indirect=["valid_vrtool_config"],
    )
    def test_run_optimization(self, valid_vrtool_config: VrtoolConfig):
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

        self._validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        _acceptance_measure_test_cases,
        indirect=["valid_vrtool_config"],
    )
    @pytest.mark.skip(
        reason="Currently only slows down the tests, reenable when optimization can be exported, and all acceptance tests can be run"
    )
    def test_run_measures_and_save_measure_results(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _test_traject = get_dike_traject(valid_vrtool_config)
        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        _results = RunMeasures(valid_vrtool_config, _test_traject).run()

        # 3. Verify expectations.
        assert isinstance(_results, ResultsMeasures)
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # NOTE: Ideally this is done with the context manager and a db.savepoint() transaction.
        # However, this is not possible as the connection will be closed during the export_initial_assessment.
        # Causing an error as the transaction requires said connection to be open.
        # Therefore the following has been found as the only possible way to assess whether the results are
        # written in the database without affecting other tests from using this db.
        _bck_db_name = "bck_db.db"
        _bck_db_filepath = valid_vrtool_config.input_database_path.with_name(
            _bck_db_name
        )
        shutil.copyfile(valid_vrtool_config.input_database_path, _bck_db_filepath)
        _results.vr_config.input_database_name = _bck_db_name

        # 4. Validate exporting results is possible
        clear_measure_results(valid_vrtool_config)
        export_solutions(_results)
        self.validate_measure_results(valid_vrtool_config)

    def validate_measure_results(self, valid_vrtool_config: VrtoolConfig):
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
        open_database(valid_vrtool_config.input_database_path)

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

        assert len(MeasureResult.select()) == total_nr_of_measure_results
        assert (
            len(MeasureResultParameter.select())
            == total_nr_of_measure_result_parameters
        )

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

                assert _measure_result.beta == pytest.approx(
                    reference_section_reliabilities[year], 0.00000001
                ), "Mismatched beta for section {} and measure result {} with id {}, dberm {}, dcrest {}, and year {}".format(
                    section.section_name,
                    row[("type",)],
                    casted_measure_id,
                    dberm,
                    dcrest,
                    casted_year,
                )

                assert _measure_result.cost == float(
                    row[("cost",)]
                ), "Mismatched cost for section {} and measure result {} with id {}, dberm {}, dcrest {}, and year {}".format(
                    section.section_name,
                    row[("type",)],
                    casted_measure_id,
                    dberm,
                    dcrest,
                    casted_year,
                )

    def is_soil_reinforcement_measure(self, dberm: float, dcrest: float) -> bool:
        return dberm != -999 and dcrest != -999

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
            & (MeasureResult.time == int(year))
        )

        assert isinstance(
            _measure_result, MeasureResult
        ), "No entry found for section {} and measure result {} with id {} and year {}".format(
            measure_per_section.section.section_name, measure_type, measure_id, year
        )
        assert not any(_measure_result.measure_result_parameters.select())

        return _measure_result

    @pytest.mark.skip(reason="TODO. No (test) input data available.")
    def test_investments_safe(self):
        """
        Test migrated from previous tools.RunSAFE.InvestmentsSafe
        """
        ## MAKE TRAJECT OBJECT
        _test_config = VrtoolConfig()
        _test_traject = DikeTraject.from_config(_test_config)

        ## READ ALL DATA
        ##First we read all the input data for the different sections. We store these in a Traject object.
        # Initialize a list of all sections that are of relevance (these start with DV).
        _results = RunFullModel(_test_config, _test_traject).run()

        # Now some general output figures and csv's are generated:

        # First make a table of all the solutions:
        _measure_table = StrategyBase.get_measure_table(
            _results.results_solutions, language="EN", abbrev=True
        )

        ## write a LOG of all probabilities for all steps:
        _investment_steps_dir = (
            _test_config.output_directory / "results" / "investment_steps"
        )
        if not _investment_steps_dir.exists():
            _investment_steps_dir.mkdir(parents=True)
        _results.results_strategies[0].write_reliability_to_csv(
            _investment_steps_dir, type=_test_config.design_methods[0]
        )
        _results.results_strategies[1].write_reliability_to_csv(
            _investment_steps_dir, type=_test_config.design_methods[1]
        )

        for count, _result_strategy in enumerate(_results.results_strategies):
            ps = []
            for i in _result_strategy.Probabilities:
                _, p_t = calc_traject_prob(i, horizon=100)
                ps.append(p_t)
            pd.DataFrame(ps, columns=range(100)).to_csv(
                path_or_buf=_investment_steps_dir.joinpath(
                    "PfT_" + _test_config.design_methods[count] + ".csv",
                )
            )
