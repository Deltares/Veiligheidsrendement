from pathlib import Path

from geolib import DStabilityModel
from geolib.geometry import Point as GeolibPoint
from geolib.models.dstability.reinforcements import ForbiddenLine


class DStabilityWrapper:
    def __init__(self, stix_path: Path | None, externals_path: Path | None) -> None:

        if not stix_path:
            raise ValueError("Missing argument value stix_path.")

        if not externals_path:
            raise ValueError("Missing argument value externals_path.")

        self.stix_path = stix_path
        self._dstability_model = DStabilityModel()
        self._dstability_model.parse(self.stix_path)
        # We only need to provide where the "DStabilityConsole" directory is.
        # https://deltares.github.io/GEOLib/latest/user/setup.html
        self._dstability_model.set_meta_property(
            "dstability_console_path",
            externals_path.joinpath("DStabilityConsole", "D-Stability Console.exe"),
        )

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

    def get_all_stage_ids(self) -> list[tuple[int, int]]:
        """Return a list with all the stage ids as integer from the dstability model"""
        return [
            (scenario_id, stage_id)
            for scenario_id, scenario in enumerate(self._dstability_model.scenarios)
            for stage_id, _ in enumerate(scenario.Stages)
        ]

    def rerun_stix(self) -> None:
        self._dstability_model.execute()

    def get_safety_factor(self) -> float:
        """
        Get the safety factor of a DStability calculation.

        Returns:
            The safety factor of the requested stage.

        """
        _results = self._dstability_model.output[-1]
        if _results:
            return _results.FactorOfSafety

        # if no results are found, rerun the model and try again
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

        for (_scen_id, _stage_id) in self.get_all_stage_ids():
            self._dstability_model.add_reinforcement(
                _stability_screen, scenario_index=_scen_id, stage_index=_stage_id
            )
