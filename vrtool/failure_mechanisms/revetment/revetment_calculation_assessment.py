import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d
from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import StoneSlopePart
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import GrassSlopePart
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import StoneSlopePart
import numpy as np
from scipy import interpolate


class RevetmentCalculation(FailureMechanismCalculatorProtocol):
    def __init__(self, revetment: RevetmentDataClass) -> None:
        self._r = revetment

    def beta_comb(self, betaZST: list[float], betaGEBU: float) -> float:
        if np.all(np.isnan(betaZST)):
            probZST = 0.0
        else:
            probZST = norm.cdf(-np.nanmin(betaZST))

        if np.isnan(betaGEBU):
            probGEBU = 0.0
        else:
            probGEBU = norm.cdf(-betaGEBU)

        probComb = probZST + probGEBU
        betaComb = -ndtri(probComb)
        return betaComb

    def _calculate_slope_part_cost(
        self, slope_part: SlopePartProtocol, section_length: float, year: int
    ):
        # TODO: This is executed after evaluating the measure, then we will get what's the new  layer type.
        opslagfactor = 2.509
        discontovoet = 1.02

        # Opnemen en afvoeren oude steenbekleding naar verwerker (incl. stort-/recyclingskosten)
        _cost_remove_steen = 5.49

        # Opnemen en afvoeren teerhoudende oude asfaltbekleding (D=15cm) (incl. stort-/recyclingskosten)
        _cost_remove_asfalt = 13.52

        # Leveren en aanbrengen (verwerken) betonzuilen, incl. doek, vijlaag en inwassen
        D = np.array([0.3, 0.35, 0.4, 0.45, 0.5])
        cost = np.array([72.52, 82.70, 92.56, 102.06, 111.56])
        f = interpolate.interp1d(D, cost, fill_value=("extrapolate"))
        cost_new_steen = f(slope_part.top_layer_thickness)

        _slope_part_difference = slope_part.end_part - slope_part.begin_part
        x = _slope_part_difference / slope_part.tan_alpha

        if x < 0.0 or slope_part.end_part < slope_part.begin_part:
            raise ValueError("Calculation of design area not possible!")

        # calculate area of new design
        z = np.sqrt(x**2 + _slope_part_difference**2)
        area = z * section_length

        if isinstance(slope_part, StoneSlopePart):  # cost of new steen
            cost_vlak = _cost_remove_steen + cost_new_steen
        elif slope_part.top_layer_type == 2026.0:
            # cost of new steen, when previous was gras
            cost_vlak = cost_new_steen
        elif isinstance(slope_part, GrassSlopePart):
            # cost of removing old revetment when new revetment is gras
            if prev_toplaagtype == 5.0:
                cost_vlak = _cost_remove_asfalt
            elif prev_toplaagtype == 20.0:
                cost_vlak = 0.0
            else:
                cost_vlak = _cost_remove_steen
        else:
            cost_vlak = 0.0

        return area * cost_vlak * opslagfactor / discontovoet ** (year - 2025)

    def calculate_cost(self, section_length: float, year: int) -> list[float]:
        _costs = []
        for _slope_part in self._r.slope_parts:
            _costs.append(
                self._calculate_slope_part_cost(_slope_part, section_length, year)
            )
        return _costs

    def calculate(self, year: int) -> tuple[float, float]:
        # TODO: we don't actually use the year, whilst in other calculators we do.
        # We need clarification on this.
        beta_zst = []
        beta_gebu = np.nan

        for _slope_part in self._r.slope_parts:
            if isinstance(_slope_part, StoneSlopePart):
                beta_zst.append(self._evaluate_block(_slope_part))
            elif isinstance(_slope_part, GrassSlopePart) and np.isnan(beta_gebu):
                beta_zst.append(np.nan)
                beta_gebu = self._evaluate_grass()
            else:
                beta_zst.append(np.nan)

        return beta_zst, beta_gebu

    def _evaluate_block(self, slope_part: StoneSlopePart):
        D_opt = []
        betaFailure = []
        for _slope_part_relation in slope_part.slope_part_relations:
            D_opt.append(_slope_part_relation.top_layer_thickness)
            betaFailure.append(_slope_part_relation.beta)

        fBlock = interp1d(D_opt, betaFailure, fill_value=("extrapolate"))
        beta = fBlock(slope_part.top_layer_thickness)

        return beta

    def _evaluate_grass(self):
        transitions = []
        betaFailure = []
        for rel in self._r.grass_relations:
            transitions.append(rel.transition_level)
            betaFailure.append(rel.beta)

        fgrass = interp1d(transitions, betaFailure, fill_value=("extrapolate"))
        beta = fgrass(self._r.current_transition_level)

        return beta
