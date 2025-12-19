from __future__ import annotations

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection


class DikeTraject:
    general_info: DikeTrajectInfo
    sections: list[DikeSection]
    mechanisms: list[MechanismEnum]
    t_0: int
    T: list[int]

    # This class contains general information on the dike traject and is used to store all data on the sections
    @classmethod
    def from_config(cls, config: VrtoolConfig) -> DikeTraject:
        """
        Generates a `DikeTraject` from a simple `VrtoolConfig` object.

        Args:
            config (VrtoolConfig): valid `VrtoolConfig` object.

        Raises:
            ValueError: When no traject value has been provided.

        Returns:
            DikeTraject: object containing the related values from the config.
        """
        if not config.traject:
            raise ValueError("No traject given in config.")

        _dike_traject = cls()

        _dike_traject.mechanisms = config.mechanisms
        _dike_traject.t_0 = config.t_0
        _dike_traject.T = config.T

        _dike_traject.sections = DikeSection.get_dike_sections_from_vr_config(config)

        _traject_length = sum(map(lambda x: x.Length, _dike_traject.sections))
        _dike_traject.general_info = DikeTrajectInfo.from_traject_info(
            config.traject, _traject_length
        )

        return _dike_traject
