from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability


class RevetmentMeasureMechanismReliability(MechanismReliability):
    def calculate_reliability(
        self,
        strength: MechanismInput,
        load: LoadInput,
        mechanism: str,
        year: float,
        traject_info: DikeTrajectInfo,
    ):
        # TODO (VRTOOL-187).
        # This is done to prevent a Revetment mechanism to be calculated because we already know its beta combined.
        pass
