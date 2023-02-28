import logging
import shelve
from pathlib import Path

from src.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsSafetyAssessment(VrToolRunResultProtocol):
    def plot_results(self):
        """
        Plot initial reliability for `selected_traject`.
        """
        _case_settings = {
            "directory": self.vr_config.output_directory,
            "language": self.vr_config.language,
            "beta_or_prob": self.vr_config.beta_or_prob,
        }

        # Previously this plotting would be skipped for 'test' type of plotting.
        self.selected_traject.plotAssessment(
            fig_size=(12, 4),
            draw_targetbeta="off",
            last=True,
            t_list=[0, 25, 50],
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

    def save_results(self):
        # Save intermediate results to shelf:
        my_shelf = shelve.open(str(self._step_output_filepath), "n")
        my_shelf["SelectedTraject"] = self.selected_traject
        my_shelf.close()

    def load_results(self):
        if self._step_output_filepath.exists():
            _shelf = shelve.open(str(self._step_output_filepath))
            self.selected_traject = _shelf["SelectedTraject"]
            _shelf.close()
            logging.info("Loaded Selected Traject from file")

        return super().load_results()
