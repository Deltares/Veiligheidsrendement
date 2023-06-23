import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.interpolate import interp1d


class revetmentCalculation:
    def beta_comb(self, betaZST: float, betaGEBU: float):

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

    def issteen(self, toplaagtype):

        res = False

        if toplaagtype >= 26.0 and toplaagtype <= 27.9:
            res = True

        return res

    def isgras(self, toplaagtype):

        res = False

        if toplaagtype == 20.0:
            res = True

        return res

    def evaluate_steen(self, D, D_opt, betaFalen):

        fsteen = interp1d(D_opt, betaFalen, fill_value=("extrapolate"))
        beta = fsteen(D)

        return beta

    def evaluate_gras(self, h_onder, grasbekleding_begin, betaFalen):

        fgras = interp1d(grasbekleding_begin, betaFalen, fill_value=("extrapolate"))
        beta = fgras(h_onder)

        return beta

    def evaluate_bekleding(self, dataZST, dataGEBU):

        betaZST = []
        betaGEBU = []

        for i in range(0, dataZST["aantal deelvakken"]):

            if self.issteen(dataZST["toplaagtype"][i]):  # for steen

                betaZST.append(
                    self.evaluate_steen(
                        dataZST["D huidig"][i],
                        dataZST[f"deelvak {i}"]["D_opt"],
                        dataZST[f"deelvak {i}"]["betaFalen"],
                    )
                )
                betaGEBU.append(np.nan)

            elif self.isgras(dataZST["toplaagtype"][i]):  # for gras

                betaZST.append(np.nan)
                betaGEBU.append(
                    self.evaluate_gras(
                        dataZST["overgang huidig"],
                        dataGEBU["grasbekleding_begin"],
                        dataGEBU["betaFalen"],
                    )
                )

            else:

                betaZST.append(np.nan)
                betaGEBU.append(np.nan)

        return betaZST, betaGEBU
