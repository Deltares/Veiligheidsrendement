import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ndtri

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    SlopePartProtocol,
    StoneSlopePart,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class RevetmentCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, revetment: RevetmentDataClass) -> None:
        self._revetment = revetment

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
        f = interp1d(D, cost, fill_value=("extrapolate"))
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
        given_years = self._revetment.find_given_years()
        betaPerYear = []
        for given_year in given_years:
            beta_zst = []
            beta_gebu = np.nan
            for _slope_part in self._revetment.slope_parts:
                if isinstance(_slope_part, StoneSlopePart):
                    beta_zst.append(self._evaluate_block(_slope_part, given_year))
                elif isinstance(_slope_part, GrassSlopePart) and np.isnan(beta_gebu):
                    beta_zst.append(np.nan)
                    beta_gebu = self._evaluate_grass(given_year)
                else:
                    beta_zst.append(np.nan)
            betaPerYear.append(self._beta_comb(beta_zst, beta_gebu))

        if len(given_years) == 1:
            return betaPerYear[0], beta_to_pf(betaPerYear[0])
        else:
            intBeta = interp1d(given_years, betaPerYear, fill_value=("extrapolate"))
            finalBeta = intBeta(year)
            return finalBeta, beta_to_pf(finalBeta)

    def _beta_comb(self, betaZST: list[float], betaGEBU: float) -> float:
        if np.all(np.isnan(betaZST)):
            probZST = 0.0
        else:
            probZST = beta_to_pf(np.nanmin(betaZST))

        if np.isnan(betaGEBU):
            probGEBU = 0.0
        else:
            probGEBU = beta_to_pf(betaGEBU)

        probComb = probZST + probGEBU
        betaComb = -ndtri(probComb)
        return betaComb

    def _evaluate_block(self, slope_part: StoneSlopePart, given_year: int):
        D_opt = []
        betaFailure = []
        for _slope_part_relation in slope_part.slope_part_relations:
            if _slope_part_relation.year == given_year:
                D_opt.append(_slope_part_relation.top_layer_thickness)
                betaFailure.append(_slope_part_relation.beta)

        fBlock = interp1d(D_opt, betaFailure, fill_value=("extrapolate"))
        beta = fBlock(slope_part.top_layer_thickness)

        return beta

    def _evaluate_grass(self, given_year: int):
        transitions = []
        betaFailure = []
        for rel in self._revetment.grass_relations:
            if rel.year == given_year:
                transitions.append(rel.transition_level)
                betaFailure.append(rel.beta)

        fgrass = interp1d(transitions, betaFailure, fill_value=("extrapolate"))
        beta = fgrass(self._revetment.current_transition_level)

        return beta
