from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)

from vrtool.common.measure_unit_costs import MeasureUnitCosts


@dataclass
class RevetmentMeasureData:
    begin_part: float
    end_part: float
    top_layer_type: float
    previous_top_layer_type: float
    top_layer_thickness: float
    beta_block_revetment: float
    beta_grass_revetment: float
    reinforce: bool
    tan_alpha: float

    def get_total_cost(self, section_length: float, unit_costs: MeasureUnitCosts) -> float:
        """
        Calculates the associated costs of this `RevetmentMeasureData` for a given dike section length (`section_length`).

        Args:
            section_length (float): Length of the Dike section whose costs will be calculated.

        Raises:
            ValueError: When the design are cannot be calculated (`end_part` < `begin_part` or negative slope).

        Returns:
            float: Total of related costs (without a specified unit, assume Euros).
        """
        if not self.reinforce:
            return 0.0

        # Leveren en aanbrengen (verwerken) betonzuilen, incl. doek, vijlaag en inwassen
        # Make sure values are sorted (check for consistency is done with import of unit_costs)

        _block_revetment_installation = interp1d([key/100 for key in sorted(unit_costs.installation_of_blocks.keys())], 
                                                 sorted(list(unit_costs.installation_of_blocks.values())), 
                                                 fill_value=("extrapolate"))

        cost_new_steen = _block_revetment_installation(self.top_layer_thickness)

        _slope_part_difference = self.end_part - self.begin_part
        x = _slope_part_difference / self.tan_alpha

        if x < 0.0 or self.end_part < self.begin_part:
            raise ValueError("Calculation of design area not possible!")

        # calculate area of new design
        z = np.sqrt(x**2 + _slope_part_difference**2)
        area = z * section_length

        if StoneSlopePart.is_stone_slope_part(self.top_layer_type):  # cost of new steen
            cost_vlak = unit_costs.remove_block_revetment + cost_new_steen
        elif self.top_layer_type == 2026.0:
            # cost of new steen, when previous was gras
            cost_vlak = cost_new_steen
        elif GrassSlopePart.is_grass_part(self.top_layer_type):
            # cost of removing old revetment when new revetment is grass
            if 1.0 <= self.previous_top_layer_type <= 5.0:
                cost_vlak = unit_costs.remove_asphalt_revetment
            elif self.previous_top_layer_type == 20.0:
                cost_vlak = 0.0
            else:
                cost_vlak = unit_costs.remove_block_revetment
        else:
            cost_vlak = 0.0

        return area * cost_vlak
