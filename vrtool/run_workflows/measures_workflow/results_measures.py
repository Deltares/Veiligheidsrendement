from __future__ import annotations

from pathlib import Path

from vrtool.decision_making.solutions import Solutions
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: dict[str, Solutions]
    ids_to_import: list[tuple[int, int]]

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
