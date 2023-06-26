import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePart
from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart


class revetmentCalculation:
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

    def evaluate_assessment(self):
        beta_zst = []
        beta_gebu = np.nan

        for _slope_part in self._r.slope_parts:
            if isinstance(_slope_part, StoneSlopePart):
                beta_zst.append(self._evaluate_block(_slope_part.top_layer_thickness))
            elif isinstance(_slope_part, GrassSlopePart) and np.isnan(beta_gebu):
                beta_zst.append(np.nan)
                beta_gebu = self._evaluate_grass()
            else:
                beta_zst.append(np.nan)

        return beta_zst, beta_gebu

    def _evaluate_block(self, slope_part: SlopePart):
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
