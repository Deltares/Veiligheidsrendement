from __future__ import annotations

import logging

from tqdm import tqdm

from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.orm_controllers import get_dike_section_solutions
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunMeasures(VrToolRunProtocol):
    def __init__(
        self,
        vr_config: VrtoolConfig,
        selected_traject: DikeTraject,
    ) -> None:
        if not isinstance(vr_config, VrtoolConfig):
            raise ValueError("Expected instance of a {}.".format(VrtoolConfig.__name__))
        if not isinstance(selected_traject, DikeTraject):
            raise ValueError("Expected instance of a {}.".format(DikeTraject.__name__))

        self.vr_config = vr_config
        self.selected_traject = selected_traject

    def _get_section_solution(
        self,
        selected_section: DikeSection,
    ) -> tuple[str, Solutions]:
        _solution = get_dike_section_solutions(
            self.vr_config, selected_section, self.selected_traject.general_info
        )
        return selected_section.name, _solution

    def run(self) -> ResultsMeasures:
        # Get measures solutions
        logging.info("Start stap 2: bepaling effecten en kosten van maatregelen.")
        _results_measures = ResultsMeasures()
        _results_measures.vr_config = self.vr_config
        _results_measures.selected_traject = self.selected_traject
        _results_measures.solutions_dict.update(
            dict(
                map(
                    self._get_section_solution,
                    tqdm(
                        self.selected_traject.sections,
                        desc="Aantal doorgerekende dijkvakken: ",
                    ),
                )
            )
        )

        for i in self.selected_traject.sections:
            _results_measures.solutions_dict[i.name].solutions_to_dataframe(
                filtering=False, splitparams=True
            )

        logging.info("Stap 2: Bepaling effecten en kosten van maatregelen afgerond.")

        return _results_measures
