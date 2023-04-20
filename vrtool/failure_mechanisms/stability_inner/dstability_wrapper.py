from pathlib import Path
from typing import Optional
from geolib import DStabilityModel

class DStabilityWrapper:
    def __init__(self, stix_path: Path, externals_location: Path) -> None:
        self.dstability_model = DStabilityModel()
        self.dstability_model.parse(stix_path)
        self._dstability_model.meta.console_folder = externals_location / "DStabilityBinaries"

    def rerun_stix(self) -> None:
        self._dstability_model.execute()

    def get_safety_factor(self, stage_id_result: Optional[int]) -> float:
        if stage_id_result is None:
            return self._dstability_model.output[-1].FactorOfSafety

        _result_id = self._dstability_model.datastructure.stages[
            stage_id_result
        ].ResultId
        if _result_id is None:
            raise Exception(
                f"The requested stage id {_result_id} does not have saved results in the provided stix {_stix_path.parts[-1]}, please rerun DStability"
            )

        for stage_output in self._dstability_model.output:
            if stage_output is not None and stage_output.Id == _result_id:
                return stage_output.FactorOfSafety