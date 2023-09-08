import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from peewee import SqliteDatabase, fn
from pandas.testing import assert_frame_equal

from tests import get_test_results_dir, test_data, test_externals
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import (
    get_dike_traject,
    open_database,
    export_results_safety_assessment,
    vrtool_db,
)
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
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
    pytest.param(
        ("TestCase1_38-1_no_housing", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, no housing",
    ),
    pytest.param(
        ("TestCase3_38-1_small", "38-1", _available_mechanisms[:3]),
        id="Traject 38-1, two sections",
    ),
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
            except:
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
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)

        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _traject
        _test_config.mechanisms = _mechanisms
        _test_config.externals = test_externals
        _test_config.input_database_path = _test_input_directory.joinpath(
            "vrtool_input.db"
        )

        assert (
            _test_config.input_database_path.exists()
        ), "No database found at {}.".format(_test_config.input_database_path)

        yield _test_config

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
        RunFullModel(valid_vrtool_config, _test_traject, VrToolPlotMode.STANDARD).run()

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
        _results = RunSafetyAssessment(
            valid_vrtool_config, _test_traject, VrToolPlotMode.STANDARD
        ).run()

        # 3. Verify expectations.
        assert isinstance(_results, ResultsSafetyAssessment)
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # NOTE: Ideally this is done with the context manager and a db.savepoint() transaction.
        # However, this is not possible as the connection will be closed during the export_initial_assessment.
        # Causing an error as the transaction requires said connection to be open.
        # Therefore the following has been found as the only possible way to assess whether the results are
        # written in the database without affecting other tests from using this db.
        _bck_db_filepath = valid_vrtool_config.output_directory.joinpath("bck_db.db")
        shutil.copyfile(valid_vrtool_config.input_database_path, _bck_db_filepath)
        _results.vr_config.input_database_path = _bck_db_filepath

        # 4. Validate exporting results is possible
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
        _results_optimization = RunOptimization(
            _results_measures, valid_vrtool_config
        ).run()

        self._validate_acceptance_result_cases(
            valid_vrtool_config.output_directory, _test_reference_path
        )

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
        _results = RunFullModel(
            _test_config, _test_traject, VrToolPlotMode.STANDARD
        ).run()

        # Now some general output figures and csv's are generated:

        # First make a table of all the solutions:
        _measure_table = StrategyBase.get_measure_table(
            _results.results_solutions, language="EN", abbrev=True
        )

        # plot beta costs for t=0
        figure_size = (12, 7)

        # LCC-beta for t = 0
        plt.figure(101, figsize=figure_size)
        _results.results_strategies[0].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            fig_id=101,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            fig_id=101,
            series_name=_test_config.design_methods[1],
            MeasureTable=_measure_table,
            last="yes",
        )
        plt.savefig(
            _test_config.output_directory.joinpath(
                "Priority order Beta vs LCC_" + str(_test_config.t_0) + ".png"
            ),
            dpi=300,
            bbox_inches="tight",
            format="png",
        )

        # LCC-beta for t=50
        plt.figure(102, figsize=figure_size)
        _results.results_strategies[0].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            t=50,
            fig_id=102,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            t=50,
            fig_id=102,
            series_name=_test_config.design_methods[1],
            MeasureTable=_measure_table,
            last="yes",
        )
        plt.savefig(
            _test_config.output_directory.joinpath(
                "Priority order Beta vs LCC_" + str(_test_config.t_0 + 50) + ".png"
            ),
            dpi=300,
            bbox_inches="tight",
            format="png",
        )

        # Costs2025-beta
        plt.figure(103, figsize=figure_size)
        _results.results_strategies[0].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            cost_type="Initial",
            fig_id=103,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _test_traject,
            save_dir=_test_config.output_directory,
            cost_type="Initial",
            fig_id=103,
            series_name=_test_config.design_methods[1],
            MeasureTable=_measure_table,
            last="yes",
        )
        plt.savefig(
            _test_config.output_directory.joinpath(
                "Priority order Beta vs Costs_" + str(_test_config.t_0 + 50) + ".png"
            ),
            dpi=300,
            bbox_inches="tight",
            format="png",
        )

        _results.results_strategies[0].plot_investment_limit(
            _test_traject,
            investmentlimit=20e6,
            path=_test_config.output_directory.joinpath("figures"),
            figure_size=(12, 6),
            years=[0],
            flip=True,
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
                beta_t, p_t = calc_traject_prob(i, horizon=100)
                ps.append(p_t)
            pd.DataFrame(ps, columns=range(100)).to_csv(
                path_or_buf=_investment_steps_dir.joinpath(
                    "PfT_" + _test_config.design_methods[count] + ".csv",
                )
            )
