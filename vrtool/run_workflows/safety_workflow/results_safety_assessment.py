from pathlib import Path

from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsSafetyAssessment(VrToolRunResultProtocol):
    def _write_results_to_file(self):
        _case_settings = {
            "directory": self.vr_config.output_directory,
            "language": self.vr_config.language,
        }

        if not self.vr_config.output_directory.exists():
            self.vr_config.output_directory.mkdir(parents=True)

        self.selected_traject.write_initial_assessment_results(
            case_settings=_case_settings,
        )

    @property
    def _step_output_filepath(self) -> Path:
        """
        Internal property to define where is located the output for the Safety Assessment step.

        Returns:
            Path: Instance representing the file location.
        """
        return self.vr_config.output_directory / "AfterStep1.out"
