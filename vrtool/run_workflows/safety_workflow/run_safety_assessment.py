import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunSafetyAssessment(VrToolRunProtocol):
    _plot_mode: VrToolPlotMode

    def __init__(
        self,
        vr_config: VrtoolConfig,
        selected_traject: DikeTraject,
        plot_mode: VrToolPlotMode,
    ) -> None:
        if not isinstance(vr_config, VrtoolConfig):
            raise ValueError("Expected instance of a {}.".format(VrtoolConfig.__name__))
        if not isinstance(selected_traject, DikeTraject):
            raise ValueError("Expected instance of a {}.".format(DikeTraject.__name__))
        self.vr_config = vr_config
        self.selected_traject = selected_traject
        self._plot_mode = plot_mode

    def run(self) -> ResultsSafetyAssessment:
        ## STEP 1: SAFETY ASSESSMENT
        logging.info("Start step 1: safety assessment")

        # Loop over sections and do the assessment.
        for _, _section in enumerate(self.selected_traject.sections):
            # get design water level:
            # TODO remove this line?
            # section.Reliability.Load.NormWaterLevel = pb.getDesignWaterLevel(section.Reliability.Load,selected_traject.GeneralInfo['Pmax'])

            # compute reliability in time for each mechanism:
            # logging.info(section.End)
            for mechanism_name in self.selected_traject.mechanism_names:
                _section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism_name
                ).generate_LCR_profile(
                    _section.section_reliability.load,
                    self.selected_traject.general_info,
                )

            # aggregate to section reliability:
            _section.section_reliability.calculate_section_reliability()

            # optional: plot reliability in time for each section
            if self.vr_config.plot_reliability_in_time:
                self._plot_reliability_in_time(_section)

        # aggregate computed initial probabilities to DataFrame in selected_traject:
        self.selected_traject.set_probabilities()

        _results = ResultsSafetyAssessment()
        _results.selected_traject = self.selected_traject
        _results.vr_config = self.vr_config
        _results.plot_results()

        logging.info("Finished step 1: assessment of current situation")
        if self.vr_config.shelves:
            _results.save_results()
        return _results

    def _get_valid_output_dir(self, path_args: list[str]) -> Path:
        _section_figures_dir = self.vr_config.output_directory.joinpath(*path_args)
        if not _section_figures_dir.exists():
            _section_figures_dir.mkdir(parents=True, exist_ok=True)
        return _section_figures_dir

    def _plot_reliability_in_time(self, selected_section: DikeSection):
        # if vr_config.plot_reliability_in_time:
        # Plot the initial reliability-time:
        plt.figure(1)
        [
            selected_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                mechanism_name
            ).drawLCR(
                mechanism=mechanism_name
            )
            for mechanism_name in self.vr_config.mechanisms
        ]
        plt.plot(
            [self.vr_config.t_0, self.vr_config.t_0 + np.max(self.vr_config.T)],
            [
                pf_to_beta(self.selected_traject.general_info.Pmax),
                pf_to_beta(self.selected_traject.general_info.Pmax),
            ],
            "k--",
            label="Norm",
        )
        plt.legend()
        plt.title(selected_section.name)
        _plot_filename = (
            self._get_valid_output_dir(["figures", selected_section.name, "Initial"])
            / "InitialSituation.png"
        )
        plt.savefig(
            _plot_filename,
            bbox_inches="tight",
        )
        plt.close()
