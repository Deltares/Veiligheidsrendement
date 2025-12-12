import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection


class Solutions:
    # This class contains possible solutions/measures
    section_name: str
    length: float
    initial_geometry: pd.DataFrame
    config: VrtoolConfig
    T: list[int]
    mechanisms: list[MechanismEnum]
    measures: list[MeasureProtocol]

    def __init__(self, dike_section: DikeSection, config: VrtoolConfig):
        self.section_name = dike_section.name
        self.length = dike_section.Length
        self.initial_geometry = dike_section.InitialGeometry

        self.config = config
        self.T = config.T
        self.mechanisms = config.mechanisms
        self.measures: list[MeasureProtocol] = []

    def _is_stability_screen_measure_valid(self) -> bool:
        return MechanismEnum.STABILITY_INNER in self.mechanisms

    def _is_soil_reinforcement_measure_valid(self, stability_screen: str) -> bool:
        if stability_screen.lower().strip() == "yes":
            return self._is_stability_screen_measure_valid()

        return True

    def evaluate_solutions(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        """This is the base routine to evaluate (i.e., determine costs and reliability) for each defined measure.
        It also gathers those measures for which availability is set to 0 and removes these from the list of measures.
        """
        for measure in self.measures:
            measure.evaluate_measure(dike_section, traject_info, preserve_slope)
