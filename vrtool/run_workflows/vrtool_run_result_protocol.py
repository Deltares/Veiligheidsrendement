from typing import Protocol, runtime_checkable

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject


@runtime_checkable
class VrToolRunResultProtocol(Protocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject
