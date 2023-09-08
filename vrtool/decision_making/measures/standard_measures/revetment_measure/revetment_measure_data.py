from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)


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

    def get_total_cost(self, section_length: float) -> float:
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
        _storage_factor = 1.000

        # Opnemen en afvoeren oude steenbekleding naar verwerker (incl. stort-/recyclingskosten)
        _cost_remove_steen = 15.66

        # Opnemen en afvoeren teerhoudende oude asfaltbekleding (D=15cm) (incl. stort-/recyclingskosten)
        _cost_remove_asfalt = 13.52 * 2.509  # TODO update!

        # Leveren en aanbrengen (verwerken) betonzuilen, incl. doek, vijlaag en inwassen
        D = np.array([0.3, 0.35, 0.4, 0.45, 0.5])
        cost = np.array([206.89, 235.93, 264.06, 291.16, 318.26])
        f = interp1d(D, cost, fill_value=("extrapolate"))
        cost_new_steen = f(self.top_layer_thickness)

        _slope_part_difference = self.end_part - self.begin_part
        x = _slope_part_difference / self.tan_alpha

        if x < 0.0 or self.end_part < self.begin_part:
            raise ValueError("Calculation of design area not possible!")

        # calculate area of new design
        z = np.sqrt(x**2 + _slope_part_difference**2)
        area = z * section_length

        if StoneSlopePart.is_stone_slope_part(self.top_layer_type):  # cost of new steen
            cost_vlak = _cost_remove_steen + cost_new_steen
        elif self.top_layer_type == 2026.0:
            # cost of new steen, when previous was gras
            cost_vlak = cost_new_steen
        elif GrassSlopePart.is_grass_part(self.top_layer_type):
            # cost of removing old revetment when new revetment is gras
            if self.previous_top_layer_type == 5.0:
                cost_vlak = _cost_remove_asfalt
            elif self.previous_top_layer_type == 20.0:
                cost_vlak = 0.0
            else:
                cost_vlak = _cost_remove_steen
        else:
            cost_vlak = 0.0

        return area * cost_vlak * _storage_factor
