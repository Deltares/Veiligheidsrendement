import logging
import shelve
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem import DikeTraject
from src.FloodDefenceSystem.DikeSection import DikeSection
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
)


class ResultsSafetyAssessment(VrToolRunResultProtocol):
    selected_traject: DikeTraject
    vr_config: VrtoolConfig

    def plot_results(self):
        """
        Plot initial reliability for `selected_traject`.
        """
        _case_settings = {
            "directory": self.vr_config.output_directory,
            "language": self.vr_config.language,
            "beta_or_prob": self.vr_config.beta_or_prob,
        }

        # Previously this plotting would be skipped for 'test' type of plotting.
        self.selected_traject.plotAssessment(
            fig_size=(12, 4),
            draw_targetbeta="off",
            last=True,
            t_list=[0, 25, 50],
            case_settings=_case_settings,
        )

    def save_results(self):
        # Save intermediate results to shelf:
        _output_file = self.vr_config.output_directory.joinpath("AfterStep1.out")
        my_shelf = shelve.open(str(_output_file), "n")
        my_shelf["SelectedTraject"] = self.selected_traject
        my_shelf.close()

    def load_results(self):
        _step_one_results = self.vr_config.output_directory.joinpath("AfterStep1.out")
        if _step_one_results.exists():
            _shelf = shelve.open(str(_step_one_results))
            self.selected_traject = _shelf["SelectedTraject"]
            _shelf.close()
            logging.info("Loaded Selected Traject from file")

        return super().load_results()


class RunSafetyAssessment(VrToolRunProtocol):
    def __init__(self, plot_mode: VrToolPlotMode) -> None:
        self._plot_mode = plot_mode

    def run(self) -> ResultsSafetyAssessment:
        ## STEP 1: SAFETY ASSESSMENT
        logging.info("Start step 1: safety assessment")

        # Loop over sections and do the assessment.
        for _, _section in enumerate(self.selected_traject.Sections):
            # get design water level:
            # TODO remove this line?
            # section.Reliability.Load.NormWaterLevel = pb.getDesignWaterLevel(section.Reliability.Load,selected_traject.GeneralInfo['Pmax'])

            # compute reliability in time for each mechanism:
            # logging.info(section.End)
            for j in self.selected_traject.GeneralInfo["MechanismsConsidered"]:
                _section.Reliability.Mechanisms[j].generateLCRProfile(
                    _section.Reliability.Load,
                    mechanism=j,
                    trajectinfo=self.selected_traject.GeneralInfo,
                )

            # aggregate to section reliability:
            _section.Reliability.calcSectionReliability()

            # optional: plot reliability in time for each section
            if self.vr_config.plot_reliability_in_time:
                self._plot_reliability_in_time(_section)

        # aggregate computed initial probabilities to DataFrame in selected_traject:
        self.selected_traject.setProbabilities()

        _results = ResultsSafetyAssessment()
        _results.selected_traject = self.selected_traject
        _results.vr_config = self.vr_config
        _results.plot_results()

        logging.info("Finished step 1: assessment of current situation")
        if self.vr_config.shelves:
            _results.save_results()
        return _results

    def _get_valid_output_dir(self, path_args: List[str]) -> Path:
        _section_figures_dir = self.vr_config.output_directory.joinpath(*path_args)
        if not _section_figures_dir.exists():
            _section_figures_dir.mkdir(parents=True, exist_ok=True)
        return _section_figures_dir

    def _plot_reliability_in_time(self, selected_section: DikeSection):
        # if vr_config.plot_reliability_in_time:
        # Plot the initial reliability-time:
        plt.figure(1)
        [
            selected_section.Reliability.Mechanisms[j].drawLCR(mechanism=j)
            for j in self.vr_config.mechanisms
        ]
        plt.plot(
            [self.vr_config.t_0, self.vr_config.t_0 + np.max(self.vr_config.T)],
            [
                pb.pf_to_beta(self.selected_traject.GeneralInfo["Pmax"]),
                pb.pf_to_beta(self.selected_traject.GeneralInfo["Pmax"]),
            ],
            "k--",
            label="Norm",
        )
        plt.legend()
        plt.title(selected_section.name)
        _plot_filename = self._get_valid_output_dir(
            ["figures", selected_section.name, "Initial", "InitialSituation.png"]
        )
        plt.savefig(
            _plot_filename,
            bbox_inches="tight",
        )
        plt.close()
