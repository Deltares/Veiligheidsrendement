from __future__ import annotations

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.optimization_workflow.run_optimization import RunOptimization
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from src.run_workflows.vrtool_run_protocol import VrToolRunProtocol
from src.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
import logging

"""
!!IMPORTANT!!
This file is deprecated in favor of the /run_workflows module which should be used instead.
This is just a 'newer' representation of what used to be in /tools/RunModel.py
Use the contents of this file for reference purposes.
"""

class RunFullModel(VrToolRunProtocol):
    def __init__(self, vr_config: VrtoolConfig, selected_traject: DikeTraject, plot_mode: VrToolPlotMode) -> None:
        if not isinstance(vr_config, VrtoolConfig):
            raise ValueError("Expected instance of a {}.".format(VrtoolConfig.__name__))
        if not isinstance(selected_traject, DikeTraject):
            raise ValueError("Expected instance of a {}.".format(DikeTraject.__name__))

        self.vr_config = vr_config
        self.selected_traject = selected_traject
        self._plot_mode = plot_mode

    def run(self) -> ResultsOptimization:
        """This is the main routine for a "SAFE"-type calculation
        Input is a TrajectObject = DikeTraject object with all relevant data
        plot_mode sets the amount of plots to be made. 'test' means a simple test approach where only csv's are given as output.
        'standard' means that normal plots are made, and with 'extensive' all plots can be switched on (not recommended)"""
        # Make a few dirs if they dont exist yet:
        if not self.vr_config.output_directory.is_dir():
            logging.info("Creating output directories at {}".format(self.vr_config.output_directory))
            self.vr_config.output_directory.mkdir(parents=True, exist_ok=True)
            self.vr_config.output_directory.joinpath("figures").mkdir(parents=True, exist_ok=True)
            self.vr_config.output_directory.joinpath("results", "investment_steps").mkdir(
                parents=True, exist_ok=True
            )
        logging.info("Start run full model.")
        # Step 1. Safety assessment.
        _safety_assessment = RunSafetyAssessment(self.vr_config, self.selected_traject, self._plot_mode)
        _safety_assessment.run()

        # Step 2. Measures.
        _measures = RunMeasures(self.vr_config, self.selected_traject, self._plot_mode)
        _measures_result = _measures.run()

        # Step 3. Optimization.
        _optimization = RunOptimization(_measures_result, self._plot_mode)
        _optimization_result = _optimization.run()

        logging.info("Finished run full model.")
        return _optimization_result