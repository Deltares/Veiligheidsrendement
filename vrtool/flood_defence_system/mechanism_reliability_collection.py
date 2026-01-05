from dataclasses import dataclass, field

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.orm.models import computation_type
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


@dataclass
class MechanismReliabilityCollection:
    """
    Represents a collection of MechanismReliability objects over time.
    """

    mechanism: MechanismEnum
    computation_type: ComputationTypeEnum
    T: list[int] = field(default_factory=list)
    t_0: int = 0
    reliability: dict[int, MechanismReliability] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize and make collection of MechanismReliability objects
        # mechanism, type, years are universal.
        for _computation_year in self.T:
            self.reliability[_computation_year] = MechanismReliability(
                self.mechanism, computation_type, self.t_0
            )

    def generate_LCR_profile(self, load: LoadInput, traject_info: DikeTrajectInfo):
        """Generates the LifeCycleReliability profile.

        Args:
            load (LoadInput): The load input.
            traject_info (DikeTrajectInfo): The object containing the traject info.

        Raises:
            ValueError: Raised when an invalid load is provided.
        """
        # this function generates life-cycle reliability based on the years that have been calculated (so reliability in time)
        if not load:
            raise ValueError("A {} is required.".format(LoadInput.__name__))

        for _year, _reliability in self.reliability.items():
            self.reliability[_year].calculate_reliability(
                _reliability.input,
                load,
                self.mechanism,
                int(_year),
                traject_info,
            )

    def get_reliability_for_year(self, year: int) -> float | None:
        _reliability = self.reliability.get(year)
        return _reliability.pf if _reliability else None

    def set_reliability_for_year(self, year: int, pf: float) -> None:
        if year not in self.reliability:
            raise KeyError(
                f"Year {year} is not available in the reliability collection."
            )
        self.reliability[year].pf = pf
        self.reliability[year].beta = pf_to_beta(pf)
