import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass


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
        betaZST = []
        betaGEBU = np.nan

        for i in range(len(self._r.slope_parts)):

            if self._r.slope_parts[i].is_block:  # for block

                betaZST.append(
                    self._evaluate_block(self._r.slope_parts[i].top_layer_thickness, i)
                )

            elif self._r.slope_parts[i].is_grass and np.isnan(betaGEBU):  # for grass

                betaZST.append(np.nan)

                betaGEBU = self._evaluate_grass()

            else:

                betaZST.append(np.nan)

        return betaZST, betaGEBU

    def _evaluate_block(self, D: float, slopePartIndex: int):
        D_opt = []
        betaFailure = []
        for rel in self._r.block_relations:
            if rel.slope_part == slopePartIndex:
                D_opt.append(rel.top_layer_thickness)
                betaFailure.append(rel.beta)

        fBlock = interp1d(D_opt, betaFailure, fill_value=("extrapolate"))
        beta = fBlock(D)

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
