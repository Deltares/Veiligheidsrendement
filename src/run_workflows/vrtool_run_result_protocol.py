from typing import Protocol

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject


class VrToolRunResultProtocol(Protocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject

    def plot_results(self):
        pass

    def save_results(self):
        pass

    def load_results(self):
        pass
