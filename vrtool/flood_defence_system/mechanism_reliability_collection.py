import matplotlib.pyplot as plt
import openturns as ot

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability


# A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    """Represents a collection of MechanismReliability objects over time."""

    Reliability: dict[str, MechanismReliability]
    mechanism_name: str

    def __init__(
        self,
        mechanism: str,
        computation_type: str,
        computation_years: list[int],
        t_0: float,
        measure_year: int,
    ):
        """Creates a new instance of the MechanismReliabilityCollection

        Args:
            mechanism (str): The name of the mechanism.
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
        self.mechanism_name = mechanism
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

        for i in self.Reliability.keys():
            # TODO: This will iterate over all available years despite not having all the data available (revetment only has 2025 and 2050?)
            self.Reliability[i].calculate_reliability(
                self.Reliability[i].Input,
                load,
                self.mechanism_name,
                float(i),
                traject_info,
            )

    def drawLCR(self, yscale=None, type="beta", mechanism=None):
        # Draw the life cycle reliability. Default is beta but can be set to Pf
        t = []
        y = []

        for i in self.Reliability.keys():
            t.append(float(i) + self.t_0)
            if self.Reliability[i].mechanism_type == "Probabilistic":
                if self.Reliability[i].result.getClassName() == "SimulationResult":
                    y.append(
                        self.Reliability[i].result.getProbabilityEstimate()
                    ) if type == "pf" else y.append(
                        -ot.Normal().computeScalarQuantile(
                            self.Reliability[i].result.getProbabilityEstimate()
                        )
                    )
                else:
                    y.append(
                        self.Reliability[i].result.getEventProbability()
                    ) if type == "pf" else y.append(
                        self.Reliability[i].result.getHasoferReliabilityIndex()
                    )
            else:
                y.append(self.Reliability[i].Pf) if type == "pf" else y.append(
                    self.Reliability[i].Beta
                )

        plt.plot(t, y, label=mechanism)
        if yscale == "log":
            plt.yscale(yscale)

        plt.xlabel("Time")
        plt.ylabel(r"$\beta$") if type != "pf" else plt.ylabel(r"$P_f$")
        plt.title("Life-cycle reliability")
