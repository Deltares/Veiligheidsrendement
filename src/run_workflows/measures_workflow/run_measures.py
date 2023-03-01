from __future__ import annotations

import logging
from typing import Tuple

from src.DecisionMaking.Solutions import Solutions
from src.FloodDefenceSystem import DikeSection
from src.defaults.vrtool_config import VrtoolConfig
from src.run_workflows.measures_workflow.results_measures import ResultsMeasures
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from src.run_workflows.vrtool_run_protocol import VrToolRunProtocol
from src.FloodDefenceSystem.DikeTraject import DikeTraject


class RunMeasures(VrToolRunProtocol):
    def __init__(self, vr_config: VrtoolConfig, selected_traject: DikeTraject, plot_mode: VrToolPlotMode) -> None:
        if not isinstance(vr_config, VrtoolConfig):
            raise ValueError("Expected instance of a {}.".format(VrtoolConfig.__name__))
        if not isinstance(selected_traject, DikeTraject):
            raise ValueError("Expected instance of a {}.".format(DikeTraject.__name__))

        self.vr_config = vr_config
        self.selected_traject = selected_traject
        self._plot_mode = plot_mode

    def _get_section_solution(
        self,
        selected_section: DikeSection,
    ) -> Tuple[str, Solutions]:
        # Calculate per section, for each measure the cost-reliability-time relations:
        _solution = Solutions(selected_section, self.vr_config)
        _solution.fillSolutions(
            self.vr_config.input_directory.joinpath(selected_section.name + ".xlsx")
        )
        _solution.evaluateSolutions(selected_section, self.selected_traject.GeneralInfo)
        return selected_section.name, _solution

    def run(self) -> ResultsMeasures:
        # Safety Assessment run
        _safety_run = RunSafetyAssessment(self.vr_config, self.selected_traject, self._plot_mode)
        _safety_run.run()

        # Get measurements solutions
        logging.info("Start step 2: evaluation of measures.")
        _results_measures = ResultsMeasures()
        if self.vr_config.reuse_output:
            _results_measures.load_results()
        else:
            _results_measures.solutions_dict.update(
                dict(map(self._get_section_solution, self.selected_traject.Sections))
            )

        for i in self.selected_traject.Sections:
            _results_measures.solutions_dict[i.name].SolutionstoDataFrame(
                filtering="off", splitparams=True
            )

        _results_measures.selected_traject = self.selected_traject
        _results_measures.vr_config = self.vr_config

        # Store intermediate results:
        if self.vr_config.shelves:
            _results_measures.save_results()

        logging.info("Finished step 2: evaluation of measures")

        # If desired: plot beta(t)-cost for all measures at a section:
        if self.vr_config.plot_measure_reliability:
            _results_measures.plot_results()
        return _results_measures
