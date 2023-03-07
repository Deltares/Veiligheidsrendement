import numpy as np

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection

"""Important: a measure is a single type of reinforcement, so for instance a stability screen. A solution can be a COMBINATION of measures (e.g. a stability screen with a berm)"""


class MeasureBase:
    """Possible change: create subclasses for different measures to make the below code more neat. Can be done jointly with adding outward reinforcement"""

    # class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    def __init__(self, inputs, config: VrtoolConfig):
        self.parameters = {}
        for i in range(0, len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]

        self.config = config
        self.crest_step = config.crest_step
        self.berm_step = config.berm_step
        self.input_directory = config.input_directory
        self.t_0 = config.t_0
        self.geometry_plot = config.geometry_plot
        self.unit_costs = config.unit_costs

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: dict[str, any],
        preserve_slope: bool = False,
    ):
        raise Exception("define subclass of measure")