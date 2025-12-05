from __future__ import annotations

from vrtool.decision_making.solutions import Solutions
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    solutions_dict: dict[str, Solutions]
    ids_to_import: list[tuple[int, int]]

    def __init__(self) -> None:
        self.solutions_dict = {}
        self.ids_to_import = []
