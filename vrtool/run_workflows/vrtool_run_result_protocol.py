from typing import Protocol, runtime_checkable

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.DikeTraject import DikeTraject


@runtime_checkable
class VrToolRunResultProtocol(Protocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject

    def plot_results(self):
        """
        Plots the results contained by this `VrToolRunResultProtocol` instance.
        """
        pass

    def save_results(self):
        """
        Saves the results contained by this `VrToolRunResultProtocol` instance.
        """
        pass

    def load_results(self):
        """
        Loads the results contained by this `VrToolRunResultProtocol` instance.
        """
        pass
