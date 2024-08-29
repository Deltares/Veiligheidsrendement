import logging
from pathlib import Path

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunSafetyAssessment(VrToolRunProtocol):
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

    def run(self) -> ResultsSafetyAssessment:
        ## STEP 1: SAFETY ASSESSMENT
        logging.info("Start stap 1: beoordeling & projectie veiligheid")

        # Loop over sections and do the assessment.
        for _, _section in enumerate(self.selected_traject.sections):
            # get design water level:
            # TODO remove this line?
            # section.Reliability.Load.NormWaterLevel = pb.getDesignWaterLevel(section.Reliability.Load,selected_traject.GeneralInfo['Pmax'])

            # compute reliability in time for each mechanism:
            for mechanism in self.selected_traject.mechanisms:
                _mechanism_reliability_collection = _section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                )

                if _mechanism_reliability_collection:
                    _mechanism_reliability_collection.generate_LCR_profile(
                        _section.section_reliability.load,
                        self.selected_traject.general_info,
                    )

            # aggregate to section reliability:
            _section.section_reliability.calculate_section_reliability()

        # aggregate computed initial probabilities to DataFrame in selected_traject:
        self.selected_traject.set_probabilities()

        _results = ResultsSafetyAssessment()
        _results.selected_traject = self.selected_traject
        _results.vr_config = self.vr_config
        _results._write_results_to_file()

        logging.info("Stap 1 afgerond.")

        return _results

    def _get_valid_output_dir(self, path_args: list[str]) -> Path:
        _section_figures_dir = self.vr_config.output_directory.joinpath(*path_args)
        if not _section_figures_dir.exists():
            _section_figures_dir.mkdir(parents=True, exist_ok=True)
        return _section_figures_dir
