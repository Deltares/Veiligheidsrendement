import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d

from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass


class revetmentCalculation:
    def beta_comb(self, betaZST, betaGEBU):

        if np.all(np.isnan(betaZST)):
            probZST = 0.0
        else:
            probZST = norm.cdf(-np.nanmin(betaZST))

        if np.all(np.isnan(betaGEBU)):
            probGEBU = 0.0
        else:
            probGEBU = norm.cdf(-np.nanmin(betaGEBU))

        probComb = probZST + probGEBU
        betaComb = -ndtri(probComb)
        return betaComb

    def evaluate_steen(self, D, D_opt, betaFalen):

        fsteen = interp1d(D_opt, betaFalen, fill_value=("extrapolate"))
        beta = fsteen(D)

        return beta

    def evaluate_gras(self, h_onder, grasbekleding_begin, betaFalen):

        fgras = interp1d(grasbekleding_begin, betaFalen, fill_value=("extrapolate"))
        beta = fgras(h_onder)

        return beta

    def evaluate_bekleding(self, revetment: RevetmentDataClass):

        betaZST = []
        betaGEBU = []

        for i in range(len(revetment.slope_parts)):

            if revetment.slope_parts[i].is_asphalt:  # for steen

                D_opt = []
                betaFalen = []
                for rel in revetment.stone_relations:
                    if rel.slope_part == i:
                        D_opt.append(rel.top_layer_thickness)
                        betaFalen.append(rel.beta)
                betaZST.append(
                    self.evaluate_steen(
                        revetment.slope_parts[i].top_layer_thickness,
                        D_opt,
                        betaFalen,
                    )
                )
                betaGEBU.append(np.nan)

            elif revetment.slope_parts[i].is_grass:  # for gras

                betaZST.append(np.nan)
                transitions = []
                betaFalen = []
                for rel in revetment.grass_relations:
                    transitions.append(rel.transition_level)
                    betaFalen.append(rel.beta)
                betaGEBU.append(
                    self.evaluate_gras(
                        revetment.current_transition_level,
                        transitions,
                        betaFalen,
                    )
                )

            else:

                betaZST.append(np.nan)
                betaGEBU.append(np.nan)

        return betaZST, betaGEBU
