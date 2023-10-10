from __future__ import annotations

import logging
import shelve
from pathlib import Path
from typing import Dict

from vrtool.decision_making.solutions import Solutions
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: Dict[str, Solutions]

    def __init__(self) -> None:
        self.solutions_dict = {}
        self.ids_to_import = []

    @property
    def _step_output_filepath(self) -> Path:
        """
        Internal property to define where is located the output for the Measures step.

        Returns:
            Path: Instance representing the file location.
        """
        return self.vr_config.output_directory / "AfterStep2.out"

    def load_results(self, alternative_path=None):
        if self._step_output_filepath.exists():
            _shelf = shelve.open(str(self._step_output_filepath))
            self.solutions_dict = _shelf["AllSolutions"]
            _shelf.close()
            logging.info("Loaded AllSolutions from file")
        elif alternative_path != None:
            _shelf = shelve.open(str(alternative_path))
            self.solutions_dict = _shelf["AllSolutions"]
            logging.info("Loaded AllSolutions from file")

    def save_results(self):
        _shelf = shelve.open(str(self._step_output_filepath), "n")
        _shelf["AllSolutions"] = self.solutions_dict
        _shelf.close()
