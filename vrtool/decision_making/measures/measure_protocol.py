from pathlib import Path

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection

"""Important: a measure is a single type of reinforcement, so for instance a stability screen. A solution can be a COMBINATION of measures (e.g. a stability screen with a berm)"""

from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class MeasureProtocol(Protocol):
    # class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    parameters: dict
    config: VrtoolConfig
    crest_step: float
    berm_step: list[int]
    t_0: int
    unit_costs: MeasureUnitCosts
    input_directory: Path
    measures: Union[list, dict, MeasureResultCollectionProtocol]

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        """
        Evaluates a measure and sets the corresponding values in its running instance.

        Args:
            dike_section (DikeSection): Dike section which will be evaluated.
            traject_info (DikeTrajectInfo): Object containing extra information required for the evaluation.
            preserve_slope (bool): Set to false if the slope can be modified during evaluation.
        """
        pass
