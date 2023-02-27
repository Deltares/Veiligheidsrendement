from __future__ import annotations

import shelve
from enum import Enum
from pathlib import Path
from typing import List, Protocol

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject


class VrToolPlotMode(Enum):
    STANDARD = 0
    EXTENSIVE = 1


class VrToolRunResultProtocol(Protocol):
    pass


class VrToolRunProtocol(Protocol):
    vr_config: VrtoolConfig
    selected_traject: DikeTraject

    def run(self) -> VrToolRunResultProtocol:
        pass


def save_intermediate_results(filename: Path, results_dict: dict) -> None:
    """
    Saves the intermediate results using the `shelve` library.

    Args:
        filename (Path): Path where to export the results
        results_dict (dict): Dictionary of values to be exported using their respective keys.
    """
    # make shelf
    my_shelf = shelve.open(str(filename), "n")
    for key, value in results_dict.items():
        my_shelf[key] = value
    my_shelf.close()


def load_intermediate_results(filename: Path, results_keys: List[str]) -> dict:
    """
    Loads intermediate results from a provided file (`filename`) using the `shelve` library.

    Args:
        filename (Path): Path from where to load the intermediate results.
        results_keys (List[str]): List of keys to be loaded from the intermediate results.

    Returns:
        dict: Resulting dictionary of values succesfully loaded.
    """
    _shelf = shelve.open(str(filename))
    _result_dict = {
        _result_key: _shelf[_result_key]
        for _result_key in results_keys
        if _result_key in _shelf.keys()
    }
    _shelf.close()
    return _result_dict
