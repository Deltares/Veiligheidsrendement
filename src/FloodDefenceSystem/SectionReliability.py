import numpy as np
import pandas as pd

import src.ProbabilisticTools.ProbabilisticFunctions as ProbabilisticFunctions


#Class describing safety assessments of a section:
class SectionReliability:
    def __init__(self):
        self
    def calcSectionReliability(self):
        #This routine translates cross-sectional to section reliability indices

        # TODO Add optional interpolation here.
        trange = [int(i) for i in self.Mechanisms[list(self.Mechanisms.keys())[0]].Reliability.keys()]
        pf_mechanisms_time = np.zeros((len(self.Mechanisms.keys()),len(trange)))
        count = 0
        for i in self.Mechanisms.keys(): #mechanisms
            for j in range(0,len(trange)):
                if i == 'Overflow':
                    pf_mechanisms_time[count,j] = self.Mechanisms[i].Reliability[str(trange[j])].Pf
                elif i == 'StabilityInner':
                    pf = self.Mechanisms[i].Reliability[str(trange[j])].Pf
                    #underneath one can choose whether to upscale within sections or not:
                    N = 1
                    # N = length/TrajectInfo['bStabilityInner']
                    # N = TrajectInfo['aStabilityInner']*length/TrajectInfo['bStabilityInner']

                    # pf_mechanisms_time[count,j] = min(1 - (1 - pf) ** N,1./100)
                    pf_mechanisms_time[count, j] = min(1 - (1 - pf) ** N,1./2)


                elif i == 'Piping':
                    pf = self.Mechanisms[i].Reliability[str(trange[j])].Pf
                    #underneath one can choose whether to upscale within sections or not:
                    N = 1
                    # N = length/TrajectInfo['bPiping']
                    # N = TrajectInfo['aPiping'] * length / TrajectInfo['bPiping']
                    # pf_mechanisms_time[count, j] = min(1 - (1 - pf) ** N,1./100)
                    pf_mechanisms_time[count, j] = min(1 - (1 - pf) ** N,1./2)
            count += 1

        #Do we want beta or failure probability? Preferably beta as output
        beta_mech_time = pd.DataFrame(ProbabilisticFunctions.pf_to_beta(pf_mechanisms_time),
                                          columns=list(self.Mechanisms[list(self.Mechanisms.keys())[0]].Reliability.keys()),
                                          index=list(self.Mechanisms.keys()))
        beta_time = pd.DataFrame([ProbabilisticFunctions.pf_to_beta(np.sum(pf_mechanisms_time,axis=0))],
                         columns=list(self.Mechanisms[list(self.Mechanisms.keys())[0]].Reliability.keys()),
                         index=['Section'])
        self.SectionReliability = pd.concat((beta_mech_time,beta_time))
        #TODO add output as probability so we dont have to switch using scipystats all the time.