from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


@runtime_checkable
class VrToolRunProtocol(Protocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject

    def run(self) -> VrToolRunResultProtocol:
        """
        Runs a Veiligheidsrendement step based on the instances of the defined `VrtoolConfig` and `DikeTraject`.

        Returns:
            VrToolRunResultProtocol: Container of all results regarding this run.
        """
        pass
