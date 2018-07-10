from scipy.stats import norm
import DikeClasses
import numpy as np

#Function to calculate a safety factor:
def calc_gamma(mechanism,DikeSection):
    if mechanism == 'Piping' or mechanism == 'Heave' or mechanism == 'Uplift':
        Pcs = (DikeSection.TrajectInfo['Pmax'] * DikeSection.TrajectInfo['omegaPiping'] * DikeSection.TrajectInfo['bPiping']) /( DikeSection.TrajectInfo['aPiping'] * DikeSection.TrajectInfo['TrajectLength'])
        betacs = -norm.ppf(Pcs)
        betamax = -norm.ppf(DikeSection.TrajectInfo['Pmax'])
        if mechanism == 'Piping':
            gamma = 1.04*np.exp(0.37*betacs-0.43*betamax)
        elif mechanism == 'Heave':
            gamma = 0.37*np.exp(0.48*betacs-0.3*betamax)
        elif mechanism == 'Uplift':
            gamma = 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
        else:
            print('Mechanism not found')
    return gamma

#Function to calculate the implicated reliability from the safety factor
def calc_beta_implicated(mechanism,SF,DikeSection):
    if mechanism == 'Piping':
        beta = (1 / 0.37) * (np.log(SF / 1.04) + 0.43 * -norm.ppf(DikeSection.TrajectInfo['Pmax']))
    elif mechanism == 'Heave':
        beta = (1 / 0.46) * (np.log(SF / 0.48) + 0.27 * -norm.ppf(DikeSection.TrajectInfo['Pmax']))
    elif mechanism == 'Uplift':
        # print(SF)
        beta = (1 / 0.48) * (np.log(SF / 0.37) + 0.30 * -norm.ppf(DikeSection.TrajectInfo['Pmax']))
    else:
        print('Mechanism not found')
    return beta

#Calculates total probability from list of sections for a mechanism or for all mechanisms that can be found (to be programmed)
def calc_traject_prob(sections, mechanism):
    if isinstance(sections[0], float):
        traject_prob = sum(sections)
    else:
        Psections = []
        if mechanism == 'Piping':
            for i in range(0,len(sections)):
                if isinstance(sections[i].PipingAssessment.Pf,float):
                    betaCS = -norm.ppf(sections[i].PipingAssessment.Pf)
                else:
                    betaCS = max((sections[i].PipingAssessment.beta_cs_h, sections[i].PipingAssessment.beta_cs_p, sections[i].PipingAssessment.beta_cs_u))
                Psections.append(norm.cdf(-betaCS))
        traject_prob = sum(Psections)
    return traject_prob
