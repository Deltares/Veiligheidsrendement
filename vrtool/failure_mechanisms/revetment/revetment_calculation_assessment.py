import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass


class revetmentCalculation:
    def __init__(self, revetment: RevetmentDataClass) -> None:
        self.r = revetment

    def beta_comb(self, betaZST: list[float], betaGEBU: float):

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

    def evaluate_steen(self, D, i):

        D_opt = []
        betaFalen = []
        for rel in self.r.stone_relations:
            if rel.slope_part == i:
                D_opt.append(rel.top_layer_thickness)
                betaFalen.append(rel.beta)

        fsteen = interp1d(D_opt, betaFalen, fill_value=("extrapolate"))
        beta = fsteen(D)

        return beta

    def evaluate_gras(self):
        h_onder = self.r.current_transition_level
        transitions = []
        betaFalen = []
        for rel in self.r.grass_relations:
            transitions.append(rel.transition_level)
            betaFalen.append(rel.beta)

        fgras = interp1d(transitions, betaFalen, fill_value=("extrapolate"))
        beta = fgras(h_onder)

        return beta

    def evaluate_bekleding(self):

        betaZST = []
        betaGEBU = np.nan

        for i in range(len(self.r.slope_parts)):

            if self.r.slope_parts[i].is_block:  # for steen

                betaZST.append(
                    self.evaluate_steen(self.r.slope_parts[i].top_layer_thickness, i)
                )

            elif self.r.slope_parts[i].is_grass and np.isnan(betaGEBU):  # for gras

                betaZST.append(np.nan)

                betaGEBU = self.evaluate_gras()

            else:

                betaZST.append(np.nan)

        return betaZST, betaGEBU
