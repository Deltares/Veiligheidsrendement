from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsSafetyAssessment(VrToolRunResultProtocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject
