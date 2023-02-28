from __future__ import annotations

import logging
import shelve
from typing import Dict

import matplotlib.pyplot as plt

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.DecisionMaking.Solutions import Solutions
from src.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: Dict[str, Solutions]

    def __init__(self) -> None:
        self.solutions_dict = {}

    def plot_results(self):
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
        _shelf = shelve.open(str(_step_two_output), "n")
        _shelf["AllSolutions"] = self.solutions_dict
        _shelf.close()
