from pathlib import Path

from typing import Optional, List
from geolib import DStabilityModel


class DStabilityWrapper:
    def __init__(self, stix_path: Path, externals_path: Path) -> None:

        if not stix_path:
            raise ValueError("Missing argument value stix_path.")

        if not externals_path:
            raise ValueError("Missing argument value externals_path.")

        self.stix_name = stix_path.parts[-1]
        self._dstability_model = DStabilityModel()
        self._dstability_model.parse(stix_path)
        # We only need to provide where the "DStabilityConsole" directory is.
        # https://deltares.github.io/GEOLib/latest/user/setup.html
        self._dstability_model.meta.console_folder = externals_path

    @property
    def get_dstability_model(self) -> DStabilityModel:
        return self._dstability_model

    def save_dstability_model(self, new_filename: str, save_path: Path) -> None:
        """
        Serialize the dstability model to a new file with a given name at a given directory.
        Args:
            new_filename: The name of the new file.
            save_path: The path to the directory where the new file will be saved.


        Returns:
            None
        """
        self._dstability_model.serialize(save_path.joinpath(new_filename))

    def get_all_stage_ids(self) -> List[int]:
        """Return a list with all the stage ids as integer from the dstability model"""
        return [int(stage.Id) for stage in self._dstability_model.stages]

    def rerun_stix(self) -> None:
        self._dstability_model.execute()

    def get_safety_factor(self, stage_id_result: Optional[int]) -> float:
        """
        Get the safety factor of a DStability calculation.

        Args:
            stage_id_result: The stage id of the result to be returned. If None, the safety factor of the last stage is returned.

        Returns:
            The safety factor of the requested stage.

        """

        if stage_id_result is None:
            return self._dstability_model.output[-1].FactorOfSafety

        _result_id = self._dstability_model.datastructure.stages[
            stage_id_result
        ].ResultId

        if _result_id is None:
            raise ValueError(
                f"The requested stage id {_result_id} does not have saved results in the provided stix {self.stix_name}, please rerun DStability"
            )

        for stage_output in self._dstability_model.output:
            if stage_output is not None and stage_output.Id == _result_id:
                return stage_output.FactorOfSafety

        raise ValueError(f"No output found for the provided stage: {stage_id_result}.")
