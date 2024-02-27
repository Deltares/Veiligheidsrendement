import logging
from abc import abstractmethod

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject


class StrategyBase:
    """This defines a Strategy object, which can be allowed to evaluate a set of solutions/measures. There are currently 3 types:
    Greedy: a greedy optimization method is used here
    TargetReliability: a cross-sectional optimization in line with OI2014
    MixedInteger: a Mixed Integer optimization. Note that this has exponential runtime for large systems so it should not be used for more than approximately 13 sections.
    Note that this is the main class. Each type has a different subclass"""

    def __init__(self, type, config: VrtoolConfig):
        self.type = type
        self.discount_rate = config.discount_rate

        self.config = config
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self.T = config.T
        self.LE_in_section = config.LE_in_section

    def get_measure_from_index(self, index, section_order=False, print_measure=False):
        """ "Converts an index (n,sh,sg) to a printout of the measure data"""
        if not section_order:
            logging.warning(
                "Deriving section order from unordered dictionary. Might be wrong"
            )
            section_order = list(self.options_height.keys())
        section = section_order[index[0]]
        if index[1] > 0:
            sh = self.options_height[section_order[index[0]]].iloc[index[1] - 1]
        else:
            sh = "No measure"
            if index[2] > 0:
                sg = self.options_geotechnical[section_order[index[0]]].iloc[
                    index[2] - 1
                ]
                if not (
                    (sg.type.values == "Stability Screen")
                    or (sg.type.values == "Vertical Geotextile")
                ):
                    raise ValueError(
                        "Illegal combination for geotechnical option {}".format(
                            sg.type.values
                        )
                    )

            else:
                sg = "No measure"
                logging.debug("SECTION {}".format(section))
                logging.debug(
                    "No measures are taken at this section: sh and sg are: {} and {}".format(
                        sh, sg
                    )
                )
                return (section, sh, sg)
        if index[2] > 0:
            sg = self.options_geotechnical[section_order[index[0]]].iloc[index[2] - 1]

        if print_measure:
            logging.debug("SECTION {}".format(section))
            if isinstance(sh, str):
                logging.debug("There is no measure for height")
            else:
                logging.debug(
                    "The measure for height is a {} in year {} with dcrest={} meters of the class {}.".format(
                        sh["type"].values[0],
                        sh["year"].values[0],
                        sh["dcrest"].values[0],
                        sh["class"].values[0],
                    )
                )
            if (
                (sg.type.values == "Vertical Geotextile")
                or (sg.type.values == "Diaphragm Wall")
                or (sg.type.values == "Stability Screen")
                or (sg.type.values == "Custom")
            ):
                logging.debug(
                    " The geotechnical measure is a {}".format(sg.type.values[0])
                )
            elif isinstance(sg.type.values[0], list):  # VZG+Soil
                logging.debug(
                    " The geotechnical measure is a {} in year {} with a {} with dberm = {} in year {}".format(
                        sg.type.values[0][0],
                        sg.year.values[0][0],
                        sg.type.values[0][1],
                        sg.dberm.values[0],
                        sg.year.values[0][1],
                    )
                )
            elif sg.type.values == "Soil reinforcement":
                logging.debug(
                    " The geotechnical measure is a {} in year {} of class {} with dberm = {}".format(
                        sg.type.values[0],
                        sg.year.values[0],
                        sg["class"].values[0],
                        sg.dberm.values[0],
                    )
                )

        return (section, sh, sg)

    def evaluate(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        OI_horizon=50,
        splitparams=False,
        setting="fast",
    ):
        raise Exception(
            "General strategy can not be evaluated. Please make an object of the desired subclass (GreedyStrategy/MixedIntegerStrategy/TargetReliabilityStrategy"
        )

    @abstractmethod
    def get_total_lcc_and_risk(self, step_number: int) -> tuple[float, float]:
        raise NotImplementedError("Expected concrete definition in inherited class.")
