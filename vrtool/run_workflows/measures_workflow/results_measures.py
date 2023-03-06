from __future__ import annotations

import logging
import shelve
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta
from vrtool.decision_making.solutions import Solutions
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: Dict[str, Solutions]

    def __init__(self) -> None:
        self.solutions_dict = {}

    def plot_results(self):

        _figures_dir = self.vr_config.output_directory / "figures"
        if not _figures_dir.exists():
            _figures_dir.mkdir(parents=True)

        betaind_array = []
        for i in self.vr_config.T:
            betaind_array.append("beta" + str(i))

        plt_mech = ["Section", "Piping", "StabilityInner", "Overflow"]

        for i in self.selected_traject.sections:
            for betaind in betaind_array:
                for mech in plt_mech:
                    requiredbeta = pf_to_beta(
                        self.selected_traject.GeneralInfo["Pmax"]
                        * (
                            i.Length
                            / self.selected_traject.GeneralInfo["TrajectLength"]
                        )
                    )
                    plt.figure(1001)
                    self.solutions_dict[i.name].plot_beta_time_euro(
                        mechanism=mech,
                        beta_ind=betaind,
                        sectionname=i.name,
                        beta_req=requiredbeta,
                    )
                    plt.savefig(
                        _figures_dir.joinpath( i.name, "Measures", mech + "_" + betaind + ".png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(1001)
        logging.info("Finished making beta plots")

    @property
    def _step_output_filepath(self) -> Path:
        """
        Internal property to define where is located the output for the Measures step.

        Returns:
            Path: Instance representing the file location.
        """
        return self.vr_config.output_directory / "AfterStep2.out"

    def load_results(self):
        if self._step_output_filepath.exists():
            _shelf = shelve.open(str(self._step_output_filepath))
            self.solutions_dict = _shelf["AllSolutions"]
            _shelf.close()
            logging.info("Loaded AllSolutions from file")

    def save_results(self):
        _shelf = shelve.open(str(self._step_output_filepath), "n")
        _shelf["AllSolutions"] = self.solutions_dict
        _shelf.close()
