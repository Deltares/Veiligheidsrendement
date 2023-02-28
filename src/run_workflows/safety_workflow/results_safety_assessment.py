import logging
import shelve

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem import DikeTraject
from src.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsSafetyAssessment(VrToolRunResultProtocol):
    selected_traject: DikeTraject
    vr_config: VrtoolConfig

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

    def save_results(self):
        # Save intermediate results to shelf:
        _output_file = self.vr_config.output_directory.joinpath("AfterStep1.out")
        my_shelf = shelve.open(str(_output_file), "n")
        my_shelf["SelectedTraject"] = self.selected_traject
        my_shelf.close()

    def load_results(self):
        _step_one_results = self.vr_config.output_directory.joinpath("AfterStep1.out")
        if _step_one_results.exists():
            _shelf = shelve.open(str(_step_one_results))
            self.selected_traject = _shelf["SelectedTraject"]
            _shelf.close()
            logging.info("Loaded Selected Traject from file")

        return super().load_results()
