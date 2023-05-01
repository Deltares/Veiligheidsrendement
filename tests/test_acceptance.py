import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tests import get_test_results_dir, test_data, test_externals
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel

"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""
_acceptance_test_cases = [
    pytest.param("integrated_SAFE_16-3_small", "16-3"),
    pytest.param("TestCase1_38-1_no_housing", "38-1"),
    pytest.param("TestCase2_38-1_overflow_no_housing", "38-1",),
]


@pytest.mark.slow
class TestAcceptance:
    def _validate_acceptance_result_cases(
        self, test_results_dir: Path, test_reference_dir: Path
    ):
        comparison_errors = []
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]

        for file in files_to_compare:
            reference = pd.read_csv(
                test_reference_dir.joinpath("results", file), index_col=0
            )
            result = pd.read_csv(test_results_dir / file, index_col=0)
            if not reference.equals(result):
                comparison_errors.append("{} is different.".format(file))

        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )

    @pytest.mark.parametrize(
        "casename, traject",
        _acceptance_test_cases,
    )
    @pytest.mark.skip(
        reason="Duplicated in run_full_model, this should become just an integration test."
    )
    def test_run_as_sandbox(
        self, casename: str, traject: str, request: pytest.FixtureRequest
    ):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_dir = test_data / casename
        assert _test_dir.exists(), "No input data found at {}".format(_test_dir)

        _results_dir = get_test_results_dir(request) / casename
        if _results_dir.exists():
            shutil.rmtree(_results_dir)
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = _test_dir
        _vr_config.output_directory = _results_dir
        _vr_config.traject = traject
        _plot_mode = VrToolPlotMode.STANDARD

        # 2. Run test.
        # Step 0. Load Traject
        _selected_traject = DikeTraject(_vr_config)
        assert isinstance(_selected_traject, DikeTraject)

        # Step 1. Safety assessment.
        _safety_assessment = RunSafetyAssessment(
            _vr_config, _selected_traject, plot_mode=_plot_mode
        )
        _safety_result = _safety_assessment.run()
        assert isinstance(_safety_result, ResultsSafetyAssessment)

        # Step 2. Measures.
        _measures = RunMeasures(_vr_config, _selected_traject, plot_mode=_plot_mode)
        _measures_result = _measures.run()
        assert isinstance(_measures_result, ResultsMeasures)

        # Step 3. Optimization.
        _optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
        _optimization_result = _optimization.run()
        assert isinstance(_optimization_result, ResultsOptimization)

        # 3. Verify expectations.
        self._validate_acceptance_result_cases(_results_dir, _test_dir / "reference")

    @pytest.mark.parametrize(
        "casename, traject",
        _acceptance_test_cases,
    )
    def test_run_full_model(
        self, casename: str, traject: str, request: pytest.FixtureRequest
    ):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_input_directory = Path.joinpath(test_data, casename)
        assert _test_input_directory.exists()

        _test_results_directory = get_test_results_dir(request).joinpath(casename)
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)

        _test_reference_path = _test_input_directory / "reference"
        assert _test_reference_path.exists()

        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = traject
        _test_config.externals = test_externals
        _test_traject = DikeTraject(_test_config)

        # 2. Run test.
        RunFullModel(_test_config, _test_traject, VrToolPlotMode.STANDARD).run()

        # 3. Verify final expectations.
        self._validate_acceptance_result_cases(
            _test_results_directory, _test_reference_path
        )

    @pytest.mark.skip(reason="TODO. No (test) input data available.")
    def test_investments_safe(self):
        """
        Test migrated from previous tools.RunSAFE.InvestmentsSafe
        """
        ## MAKE TRAJECT OBJECT
        _test_config = VrtoolConfig()
        _test_traject = DikeTraject(_test_config)

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

    @pytest.mark.skip(
        reason="TODO. No (test) input data available. Needs to be adapted to new architecture."
    )
    def test_run_pilot(self):
        # 1. Define test data.
        _traject = "38-1"
        _vr_config_data = dict(
            shelves=True,
            reuse_output=True,
            beta_or_prob="beta",
            # Settings for step 1 :
            plot_reliability_in_time=False,
            plot_measure_reliability=False,
            flip_traject=True,
            assessment_plot_years=[0, 20, 50],
            # Settings for step 2 :
            geometry_plot=False,
            # Settings for step 3:
            design_methods=["Veiligheidsrendement", "Doorsnede-eisen"],
            # dictionary with settings for beta-cost curve:,
            beta_cost_settings={"symbols": True, "markersize": 10},
        )

        _test_config = VrtoolConfig(**_vr_config_data)
        _traject = DikeTraject(_test_config)

        # 2. Run test.
        _results = RunFullModel(_test_config, _traject, VrToolPlotMode.EXTENSIVE)
        _measure_table = StrategyBase.get_measure_table(
            _results.results_solutions, language="EN", abbrev=True
        )

        # 3. Plot results
        # plot beta costs for t=0
        figure_size = (12, 7)

        # LCC-beta for t = 0
        plt.figure(101, figsize=figure_size)
        _results.results_strategies[0].plot_beta_costs(
            _traject,
            save_dir=_test_config.output_directory,
            fig_id=101,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _traject,
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
            _traject,
            save_dir=_test_config.output_directory,
            t=50,
            fig_id=102,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _traject,
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
            _traject,
            save_dir=_test_config.output_directory,
            cost_type="Initial",
            fig_id=103,
            series_name=_test_config.design_methods[0],
            MeasureTable=_measure_table,
            color="b",
        )
        _results.results_strategies[1].plot_beta_costs(
            _traject,
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
            _traject,
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
            path=_investment_steps_dir, type=_test_config.design_methods[0]
        )
        _results.results_strategies[1].write_reliability_to_csv(
            path=_investment_steps_dir, type=_test_config.design_methods[1]
        )

        for count, Strategy in enumerate(_results.results_strategies):
            ps = []
            for i in Strategy.Probabilities:
                beta_t, p_t = calc_traject_prob(i, horizon=100)
                ps.append(p_t)
            pd.DataFrame(ps, columns=range(100)).to_csv(
                path_or_buf=_investment_steps_dir.joinpath(
                    "PfT_" + _test_config.design_methods[count] + ".csv"
                )
            )
