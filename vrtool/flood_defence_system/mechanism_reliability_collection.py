from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability


# A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    """Represents a collection of MechanismReliability objects over time."""

    Reliability: dict[str, MechanismReliability]
    mechanism: MechanismEnum

    def __init__(
        self,
        mechanism: MechanismEnum,
        computation_type: str,
        computation_years: list[int],
        t_0: float,
        measure_year: int,
    ):
        """Creates a new instance of the MechanismReliabilityCollection

        Args:
            mechanism (MechanismEnum): The mechanism.
            computation_type (str): The computation type.
            computation_years (list[int]): The collection of years to compute the reliability for.
            t_0 (float): The initial year.
            measure_year (int): The year to compute the measure for
        """

        # Initialize and make collection of MechanismReliability objects
        # mechanism, type, years are universal.
        # Measure_year is to indicate whether the reliability has to be recalculated or can be copied
        # (the latter is the case if a measure is taken later than the considered point in time)
        self.T = computation_years
        self.t_0 = t_0
        self.mechanism = mechanism
        self.Reliability = {}

        for _computation_year in computation_years:
            if measure_year > _computation_year:
                self.Reliability[str(_computation_year)] = MechanismReliability(
                    mechanism, computation_type, self.t_0, copy_or_calculate="copy"
                )
            else:
                self.Reliability[str(_computation_year)] = MechanismReliability(
                    mechanism, computation_type, self.t_0
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

        for i, _reliability in self.Reliability.items():
            self.Reliability[i].calculate_reliability(
                _reliability.Input,
                load,
                self.mechanism,
                int(i),
                traject_info,
            )
