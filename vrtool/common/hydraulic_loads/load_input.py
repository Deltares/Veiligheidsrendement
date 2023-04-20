import os
from pathlib import Path

import openturns as ot

from vrtool.probabilistic_tools.hydra_ring_scripts import design_table_openturns


class LoadInput:
    # class to store load data
    def __init__(self, section_fields):
        if "Load_2025" in section_fields:
            self.load_type = "HRING"
        elif "YearlyWLRise" in section_fields:
            self.load_type = "SAFE"

    def set_HRING_input(self, folder: Path, section_attributes: dict, gridpoints=1000):
        years = os.listdir(folder)
        self.distribution = {}
        for year in years:
            self.distribution[year] = design_table_openturns(
                folder.joinpath(
                    year, "{}.txt".format(section_attributes["Load_{}".format(year)])
                ),
                gridpoints=gridpoints,
            )

    def set_fromDesignTable(self, filelocation, gridpoints=1000):
        # Load is given by exceedence probability-water level table from Hydra-Ring
        self.distribution = design_table_openturns(filelocation, gridpoints=gridpoints)

    def set_annual_change(self, type="determinist", parameters=[0]):
        # set an annual change of the water level
        if type == "determinist":
            self.dist_change = ot.Dirac(parameters)
        elif type == "SAFE":  # specific formulation for SAFE
            self.dist_change = parameters[0]
            self.HBN_factor = parameters[1]
        elif type == "gamma":
            self.dist_change = ot.Gamma()
            self.dist_change.setParameter(ot.GammaMuSigma()(parameters))
