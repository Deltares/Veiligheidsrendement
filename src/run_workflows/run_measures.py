from __future__ import annotations

import logging
import shelve
from typing import Dict, Tuple

import matplotlib.pyplot as plt

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.DecisionMaking.Solutions import Solutions
from src.FloodDefenceSystem import DikeSection
from src.run_workflows.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
)


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: Dict[str, Solutions]

    def plot_results(
        self,
    ):
        betaind_array = []

        for i in self.vr_config.T:
            betaind_array.append("beta" + str(i))

        plt_mech = ["Section", "Piping", "StabilityInner", "Overflow"]

        for i in self.selected_traject.Sections:
            for betaind in betaind_array:
                for mech in plt_mech:
                    requiredbeta = pb.pf_to_beta(
                        self.selected_traject.GeneralInfo["Pmax"]
                        * (
                            i.Length
                            / self.selected_traject.GeneralInfo["TrajectLength"]
                        )
                    )
                    plt.figure(1001)
                    self.solutions_dict[i.name].plotBetaTimeEuro(
                        mechanism=mech,
                        beta_ind=betaind,
                        sectionname=i.name,
                        beta_req=requiredbeta,
                    )
                    plt.savefig(
                        self.vr_config.output_directory.joinpath(
                            "figures", i.name, "Measures", mech + "_" + betaind + ".png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(1001)
        logging.info("Finished making beta plots")

    def load_results(self):
        _step_two_output = self.vr_config.output_directory / "AfterStep2.out.dat"
        if _step_two_output.exists():
            _shelf = shelve.open(str(_step_two_output))
            self.solutions_dict = _shelf["AllSolutions"]
            _shelf.close()
            logging.info("Loaded AllSolutions from file")

    def save_results(self):
        _step_two_output = self.vr_config.output_directory / "AfterStep2.out"
        my_shelf = shelve.open(str(_step_two_output), "n")
        my_shelf["AllSolutions"] = self.solutions_dict
        my_shelf.close()


class RunMeasures(VrToolRunProtocol):
    def __init__(self, plot_mode: VrToolPlotMode) -> None:
        self._plot_mode = plot_mode

    def _get_section_solution(
        self,
        selected_section: DikeSection,
    ) -> Tuple[str, Solutions]:
        # Calculate per section, for each measure the cost-reliability-time relations:
        _solution = Solutions(selected_section)
        _solution.fillSolutions(self.vr_config.path.joinpath(selected_section.name))
        _solution.evaluateSolutions(selected_section, self.selected_traject.GeneralInfo)
        return selected_section.name, _solution

    def run(self) -> VrToolRunResultProtocol:
        # Safety Assessment run
        _safety_run = RunSafetyAssessment(self._plot_mode)
        _safety_run.selected_traject = self.selected_traject
        _safety_run.vr_config = self.vr_config
        _safety_results = _safety_run.run()

        # Get measurements solutions
        _results_measures = ResultsMeasures()
        _results_measures.selected_traject = self.selected_traject
        _results_measures.vr_config = self.vr_config
        if self.vr_config.reuse_output:
            _results_measures.load_results()
        else:
            _results_measures.solutions_dict.update(
                dict(map(self._get_section_solution, self.selected_traject.Sections))
            )
            # for i in self.selected_traject.Sections:
            #     _results_measures.solutions_dict[i.name] = Solutions(i)
            #     _results_measures.solutions_dict[i.name].fillSolutions(
            #         self.vr_config.path.joinpath(i.name + ".xlsx")
            #     )
            #     _results_measures.solutions_dict[i.name].evaluateSolutions(
            #         i, self.selected_traject.GeneralInfo
            #     )

        for i in self.selected_traject.Sections:
            _results_measures.solutions_dict[i.name].SolutionstoDataFrame(
                filtering="off", splitparams=True
            )

        # Store intermediate results:
        if self.vr_config.shelves:
            _results_measures.save_results()

        logging.info("Finished step 2: evaluation of measures")

        # If desired: plot beta(t)-cost for all measures at a section:
        if self.vr_config.plot_measure_reliability:
            _results_measures.plot_results()
        return _results_measures
