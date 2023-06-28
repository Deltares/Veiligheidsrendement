import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d
from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class RevetmentCalculation(FailureMechanismCalculatorProtocol):
    def __init__(self, revetment: RevetmentDataClass) -> None:
        self._r = revetment
        self.given_years = set()
        for _slope_part in revetment.slope_parts:
            if isinstance(_slope_part, StoneSlopePart):
                for rel in _slope_part.slope_part_relations:
                    self.given_years.add(rel.year)

    def _beta_comb(self, betaZST: list[float], betaGEBU: float) -> float:
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

    def calculate(self, year: int) -> tuple[float, float]:

        betaPerYear = []
        for given_year in self.given_years:
            beta_zst = []
            beta_gebu = np.nan
            for _slope_part in self._r.slope_parts:
                if isinstance(_slope_part, StoneSlopePart):
                    beta_zst.append(self._evaluate_block(_slope_part, given_year))
                elif isinstance(_slope_part, GrassSlopePart) and np.isnan(beta_gebu):
                    beta_zst.append(np.nan)
                    beta_gebu = self._evaluate_grass(given_year)
                else:
                    beta_zst.append(np.nan)
            betaPerYear.append(self._beta_comb(beta_zst, beta_gebu))

        if len(self.given_years) == 1:
            return betaPerYear[0], beta_to_pf(betaPerYear[0])
        else:
            intBeta = interp1d(
                list(self.given_years), betaPerYear, fill_value=("extrapolate")
            )
            finalBeta = intBeta(year)
            return finalBeta, beta_to_pf(finalBeta)

    def _evaluate_block(self, slope_part: StoneSlopePart, year: int):
        D_opt = []
        betaFailure = []
        for _slope_part_relation in slope_part.slope_part_relations:
            if _slope_part_relation.year == year:
                D_opt.append(_slope_part_relation.top_layer_thickness)
                betaFailure.append(_slope_part_relation.beta)

        fBlock = interp1d(D_opt, betaFailure, fill_value=("extrapolate"))
        beta = fBlock(slope_part.top_layer_thickness)

        return beta

    def _evaluate_grass(self, year: int):
        transitions = []
        betaFailure = []
        for rel in self._r.grass_relations:
            if rel.year == year:
                transitions.append(rel.transition_level)
                betaFailure.append(rel.beta)

        fgrass = interp1d(transitions, betaFailure, fill_value=("extrapolate"))
        beta = fgrass(self._r.current_transition_level)

        return beta
