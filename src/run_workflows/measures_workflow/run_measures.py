from __future__ import annotations

import logging
from typing import Tuple

from src.DecisionMaking.Solutions import Solutions
from src.FloodDefenceSystem import DikeSection
from src.run_workflows.measures_workflow.results_measures import ResultsMeasures
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from src.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunMeasures(VrToolRunProtocol):
    def __init__(self, plot_mode: VrToolPlotMode) -> None:
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
        _safety_run = RunSafetyAssessment(self._plot_mode)
        _safety_run.selected_traject = self.selected_traject
        _safety_run.vr_config = self.vr_config
        _safety_run.run()

        # Get measurements solutions
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
