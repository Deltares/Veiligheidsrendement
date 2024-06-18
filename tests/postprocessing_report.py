import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scripts.postprocessing.database_access_functions as daf
import scripts.postprocessing.database_analytics as dan
import scripts.postprocessing.generate_output as got
from vrtool.common.enums import MechanismEnum


@dataclass
class PostProcessingReport:
    """
    Dataclass that produces different figures and files based on its data structure.
    Mostly intended for testing.

    Note: If we want to make this class general available (`vrtool` project) then
    we need to adapt all the logic in the `scripts.postprocessing` subproject.
    """

    reference_db: Path
    result_db: Path
    report_dir: Path

    id_for_result: int = 1
    id_for_reference: int = 1
    has_revetment: bool = True  # Whether a revetment is present or not
    take_last: bool = False  # If True, the last step is taken, if False, the step with minimal total cost is taken
    colors: Any = field(default_factory=lambda: sns.color_palette("colorblind", 10))

    def __post_init__(self):
        if not self.report_dir.exists():
            self.report_dir.mkdir(parents=True)

    def generate_report(self):
        # Define colors.
        sns.set(style="whitegrid")

        # Get the runs to compare
        result_runs = daf.get_overview_of_runs(self.result_db)
        reference_runs = daf.get_overview_of_runs(self.reference_db)

        # This should generate a csv table?
        pd.DataFrame(result_runs + reference_runs)

        # For each run, we get the optimization steps and the index of the step with minimal total costs. This is the optimal combination of measures.
        optimization_steps = {
            "reference": daf.get_optimization_steps_for_run_id(
                self.reference_db, self.id_for_reference
            ),
            "result": daf.get_optimization_steps_for_run_id(
                self.result_db, self.id_for_result
            ),
        }

        # add total cost as sum of total_lcc and total_risk in each step
        if not self.take_last:
            considered_tc_step = {
                "reference": dan.get_minimal_tc_step(optimization_steps["reference"])
                - 1,
                "result": dan.get_minimal_tc_step(optimization_steps["result"]) - 1,
            }
        else:
            considered_tc_step = {
                "reference": len(optimization_steps["reference"]) - 1,
                "result": len(optimization_steps["result"]) - 1,
            }

        self._plot_total_cost_and_risk(optimization_steps, considered_tc_step)

        # Reading measures per step
        lists_of_measures = {
            "reference": daf.get_measures_for_run_id(
                self.reference_db, self.id_for_reference
            ),
            "result": daf.get_measures_for_run_id(self.result_db, self.id_for_result),
        }

        measures_per_step = {
            "reference": dan.get_measures_per_step_number(
                lists_of_measures["reference"]
            ),
            "result": dan.get_measures_per_step_number(lists_of_measures["result"]),
        }

        # If we want to see the failure probability per stap we first need to load the original assessment for each mechanism, and then we can compute the reliability for each step during the optimization.
        assessment_results = {"reference": {}, "result": {}}
        for mechanism in [
            MechanismEnum.OVERFLOW,
            MechanismEnum.PIPING,
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.REVETMENT,
        ]:
            if self.has_revetment or mechanism != MechanismEnum.REVETMENT:
                assessment_results["reference"][
                    mechanism
                ] = daf.import_original_assessment(self.reference_db, mechanism)
                assessment_results["result"][
                    mechanism
                ] = daf.import_original_assessment(self.result_db, mechanism)

        reliability_per_step = {
            "reference": dan.get_reliability_for_each_step(
                self.reference_db, measures_per_step["reference"]
            ),
            "result": dan.get_reliability_for_each_step(
                self.result_db, measures_per_step["result"]
            ),
        }

        # Based on these inputs we can make a stepwise_assessment based on the investments in reliability_per_step.
        stepwise_assessment = {
            "reference": dan.assessment_for_each_step(
                copy.deepcopy(assessment_results["reference"]),
                reliability_per_step["reference"],
            ),
            "result": dan.assessment_for_each_step(
                copy.deepcopy(assessment_results["result"]),
                reliability_per_step["result"],
            ),
        }

        # The next step is to derive the traject probability for each mechanism for each step using the `calculate_traject_probability_for_steps` function
        traject_prob = {
            "reference": dan.calculate_traject_probability_for_steps(
                stepwise_assessment["reference"]
            ),
            "result": dan.calculate_traject_probability_for_steps(
                stepwise_assessment["result"]
            ),
        }

        self._plot_traject_probability_for_step(traject_prob, considered_tc_step)

        #
        measures_per_section = {
            "reference": dan.get_measures_per_section_for_step(
                measures_per_step["reference"], considered_tc_step["reference"] + 1
            ),
            "result": dan.get_measures_per_section_for_step(
                measures_per_step["result"], considered_tc_step["result"] + 1
            ),
        }

        self._print_measure_result_ids(measures_per_section)

        #
        section_parameters = {"reference": {}, "result": {}}

        def _set_section_parameters(db_key: str, db_location: Path):
            for section in measures_per_section[db_key].keys():
                section_parameters[db_key][section] = []
                for measure in measures_per_section[db_key][section][0]:
                    parameters = daf.get_measure_parameters(measure, db_location)
                    parameters.update(daf.get_measure_costs(measure, db_location))
                    parameters.update(daf.get_measure_type(measure, db_location))
                    section_parameters[db_key][section].append(parameters)

        _set_section_parameters("reference", self.reference_db)
        _set_section_parameters("result", self.result_db)

        # Next we put this in a DataFrame such that we can easily compare the DataFrames of both runs.
        measure_parameters = {
            "reference": got.measure_per_section_to_df(
                measures_per_section["reference"], section_parameters["reference"]
            ),
            "result": got.measure_per_section_to_df(
                measures_per_section["result"], section_parameters["result"]
            ),
        }

        # Potentially we can export this to csv:
        measure_parameters["reference"].to_csv(
            str(self.report_dir.joinpath("reference_measures.csv"))
        )
        measure_parameters["result"].to_csv(
            str(self.report_dir.joinpath("result_measures.csv"))
        )

        # Analysis of reliability index
        # IMPORTANT: this variable sets the time at which the beta is analyzed
        t_analyzed = 0

        # get betas for each section and mechanism at t_analyzed
        betas_per_section_and_mech = {
            "result": dan.get_beta_for_each_section_and_mech_at_t(
                stepwise_assessment["result"][considered_tc_step["result"]], t_analyzed
            ),
            "reference": dan.get_beta_for_each_section_and_mech_at_t(
                stepwise_assessment["reference"][considered_tc_step["reference"]],
                t_analyzed,
            ),
        }

        # transform dicts to dataframe
        for run in betas_per_section_and_mech.keys():
            betas_per_section_and_mech[run] = (
                pd.DataFrame.from_dict(betas_per_section_and_mech[run])
                .rename_axis("section_id")
                .reset_index()
            )
            betas_per_section_and_mech[run]["run"] = run

        betas_per_section_and_mechanism = pd.concat(
            [
                betas_per_section_and_mech["result"],
                betas_per_section_and_mech["reference"],
            ],
            ignore_index=True,
        )
        betas_per_section_and_mechanism = pd.melt(
            betas_per_section_and_mechanism,
            id_vars=["section_id", "run"],
            var_name="mechanism",
            value_name="beta",
        )

        # Next we make a plot to compare the beta for both runs
        got.plot_comparison_of_beta_values(betas_per_section_and_mechanism)

        # Differences can also be revealing.
        got.plot_difference_in_betas(
            betas_per_section_and_mechanism, self.has_revetment
        )

        got.plot_difference_in_betas_per_section(
            betas_per_section_and_mechanism, self.has_revetment
        )

    def _plot_total_cost_and_risk(
        self, optimization_steps: dict, considered_tc_step: dict
    ):
        _, ax = plt.subplots()
        markers = ["d", "o"]
        for count, run in enumerate(optimization_steps.keys()):
            got.plot_lcc_tc_from_steps(
                optimization_steps[run], axis=ax, lbl=run, clr=self.colors[count]
            )
            ax.plot(
                optimization_steps[run][considered_tc_step[run]]["total_lcc"],
                optimization_steps[run][considered_tc_step[run]]["total_risk"],
                markers[count],
                color=self.colors[count],
            )
        ax.set_xlabel("Total LCC")
        ax.set_ylabel("Total risk")
        ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.set_ylim(top=1e10)
        ax.legend()

    def _plot_traject_probability_for_step(
        self, traject_prob: dict, considered_tc_step: dict
    ):
        _, ax = plt.subplots()

        got.plot_traject_probability_for_step(
            traject_prob["reference"][0],
            ax,
            run_label="Beginsituatie referentie",
            color=self.colors[0],
            linestyle="--",
        )
        got.plot_traject_probability_for_step(
            traject_prob["result"][0],
            ax,
            run_label="Beginsituatie resultaat",
            color=self.colors[1],
            linestyle=":",
        )
        got.plot_traject_probability_for_step(
            traject_prob["reference"][considered_tc_step["reference"]],
            ax,
            run_label="Referentie",
            color=self.colors[0],
            linestyle="-",
        )
        got.plot_traject_probability_for_step(
            traject_prob["result"][considered_tc_step["result"]],
            ax,
            run_label="Resultaat",
            color=self.colors[1],
            linestyle="-",
        )
        ax.set_xlim(left=0, right=100)

    def _print_measure_result_ids(self, measures_per_section: dict):
        # This should be printed to a doc?
        for section in set(
            list(measures_per_section["result"].keys())
            + list(measures_per_section["reference"].keys())
        ):
            for run_key, run_value in measures_per_section.items():
                try:
                    print(
                        f"Section {section} in run {run_key} has measures {run_value}"
                    )
                except:
                    print(
                        f"Section {section} in run {run_key} has no measures in run {run_key}"
                    )
