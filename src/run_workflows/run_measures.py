import logging

import matplotlib.pyplot as plt

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.DecisionMaking.Solutions import Solutions
from src.run_workflows.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
    load_intermediate_results,
    save_intermediate_results,
)


class RunMeasures(VrToolRunProtocol):
    def __init__(self, plot_mode: VrToolPlotMode) -> None:
        self._plot_mode = plot_mode

    def run(self) -> VrToolRunResultProtocol:
        RunSafetyAssessment(self._plot_mode).run()
        _step_two_output = self.vr_config.directory.joinpath("AfterStep2.out.dat")
        if self.vr_config.reuse_output and _step_two_output.exists():
            _results_dict = load_intermediate_results(
                _step_two_output, ["AllSolutions"]
            )
            logging.info("Loaded AllSolutions from file")
            _all_solutions = _results_dict.pop["AllSolutions"]
        else:
            _all_solutions = {}
            # Calculate per section, for each measure the cost-reliability-time relations:
            for i in self.selected_traject.Sections:
                _all_solutions[i.name] = Solutions(i)
                _all_solutions[i.name].fillSolutions(
                    self.vr_config.path.joinpath(i.name + ".xlsx")
                )
                _all_solutions[i.name].evaluateSolutions(
                    i, self.selected_traject.GeneralInfo
                )

        for i in self.selected_traject.Sections:
            _all_solutions[i.name].SolutionstoDataFrame(
                filtering="off", splitparams=True
            )

        # Store intermediate results:
        if self.vr_config.shelves:
            filename = self.vr_config.directory.joinpath("AfterStep2.out")
            save_intermediate_results(filename, dict(AllSolutions=_all_solutions))

        logging.info("Finished step 2: evaluation of measures")

        # If desired: plot beta(t)-cost for all measures at a section:
        if self.vr_config.plot_measure_reliability:
            self._plot_measure_reliability(_all_solutions)
        return _all_solutions

    def _plot_measure_reliability(
        self,
        measure_solutions: list,
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
                    measure_solutions[i.name].plotBetaTimeEuro(
                        mechanism=mech,
                        beta_ind=betaind,
                        sectionname=i.name,
                        beta_req=requiredbeta,
                    )
                    plt.savefig(
                        self.vr_config.directory.joinpath(
                            "figures", i.name, "Measures", mech + "_" + betaind + ".png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(1001)
        logging.info("Finished making beta plots")
