from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsMeasures(VrToolRunResultProtocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject
    solutions_dict: dict[str, Solutions]
    ids_to_import: list[tuple[int, int]]

    def __init__(self) -> None:
        self.solutions_dict = {}
        self.ids_to_import = []
