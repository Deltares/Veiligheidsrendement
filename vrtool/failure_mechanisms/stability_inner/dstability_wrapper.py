from pathlib import Path
from typing import Optional

from geolib import DStabilityModel
from geolib.geometry import Point as GeolibPoint
from geolib.models.dstability.reinforcements import ForbiddenLine


class DStabilityWrapper:
    def __init__(self, stix_path: Path, externals_path: Path) -> None:

        if not stix_path:
            raise ValueError("Missing argument value stix_path.")

        if not externals_path:
            raise ValueError("Missing argument value externals_path.")

        self.stix_path = stix_path
        self._dstability_model = DStabilityModel()
        self._dstability_model.parse(self.stix_path)
        # We only need to provide where the "DStabilityConsole" directory is.
        # https://deltares.github.io/GEOLib/latest/user/setup.html
        self._dstability_model.meta.console_folder = externals_path

    @property
    def get_dstability_model(self) -> DStabilityModel:
        return self._dstability_model

    def save_dstability_model(self, save_path: Path) -> None:
        """
        Serialize the dstability model to a new file with a given name at a given directory.

        Args:
            save_path: The path to the directory where the new file will be saved.

        Returns:
            None
        """
        self._dstability_model.serialize(save_path)

    def get_all_stage_ids(self) -> list[int]:
        """Return a list with all the stage ids as integer from the dstability model"""
        return [int(stage.Id) for stage in self._dstability_model.stages]

    def rerun_stix(self) -> None:
        self._dstability_model.execute()

    def get_safety_factor(self) -> float:
        """
        Get the safety factor of a DStability calculation.

        Returns:
            The safety factor of the requested stage.

        """
        try:
            _results = self._dstability_model.output[-1]
            return _results.FactorOfSafety

        # if no results are found, rerun the model and try again
        except ValueError:
            self.rerun_stix()
            _results = self._dstability_model.output[-1]
            return _results.FactorOfSafety


    def add_stability_screen(self, bottom_screen: float, location: float) -> None:
        """
        Add a stability screen to the dstability model.

        Args:
            bottom_screen: The bottom level of the stability screen.
            location: The location x of the stability screen in the D-Stability model.

        Returns:
            None
        """

        # The top of the stability screen is hardcoded at 20m to make sure it is above surface level.
        _start_screen = GeolibPoint(x=location, z=20)
        _end_screen = GeolibPoint(x=location, z=bottom_screen)
        _stability_screen = ForbiddenLine(start=_start_screen, end=_end_screen)

        for id in self.get_all_stage_ids():
            self._dstability_model.add_reinforcement(_stability_screen, stage_id=id)
