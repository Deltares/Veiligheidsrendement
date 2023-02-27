import logging

import matplotlib.pyplot as plt
import numpy as np

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.FloodDefenceSystem.DikeSection import DikeSection
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
    save_intermediate_results,
)


class RunSafetyAssessment(VrToolRunProtocol):
    def __init__(self, plot_mode: VrToolPlotMode) -> None:
        self._plot_mode = plot_mode

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
        _section_figures_dir = (
            self.vr_config.directory / "figures" / selected_section.name
        )
        if not _section_figures_dir.exists():
            _section_figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            _section_figures_dir / "Initial" / "InitialSituation.png",
            bbox_inches="tight",
        )
        plt.close()

    def run(self) -> VrToolRunResultProtocol:
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

        # Plot initial reliability for selected_traject:
        case_settings = {
            "directory": self.vr_config.directory,
            "language": self.vr_config.language,
            "beta_or_prob": self.vr_config.beta_or_prob,
        }
        # Previously this plotting would be skipped during 'test' type of plotting.
        self.selected_traject.plotAssessment(
            fig_size=(12, 4),
            draw_targetbeta="off",
            last=True,
            t_list=[0, 25, 50],
            case_settings=case_settings,
        )

        logging.info("Finished step 1: assessment of current situation")

        # store stuff:
        if self.vr_config.shelves:
            # Save intermediate results to shelf:
            save_intermediate_results(
                self.vr_config.directory.joinpath("AfterStep1.out"),
                dict(SelectedTraject=self.selected_traject),
            )
