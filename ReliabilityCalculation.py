import copy
import matplotlib.pyplot as plt
import Mechanisms
import numpy as np
import openturns as ot
import pandas as pd
import ProbabilisticFunctions
from HydraRing_scripts import DesignTableOpenTurns
from ProbabilisticFunctions import TableDist, run_prob_calc, IterativeFC_calculation, TemporalProcess, \
    addLoadCharVals, MarginalsforTimeDepReliability, beta_to_pf, pf_to_beta, FragilityIntegration
from scipy.stats import norm
from scipy import interpolate
import config
from pathlib import Path

class LoadInput:
    #class to store load data
    def __init__(self):
        pass

    def set_fromDesignTable(self, filelocation, gridpoints=1000):
        #Load is given by exceedence probability-water level table from Hydra-Ring
        self.distribution = DesignTableOpenTurns(filelocation, gridpoints=gridpoints)

    def set_annual_change(self,type='determinist',parameters = [0]):
        #set an annual change of the water level
        if type == 'determinist':
            self.dist_change = ot.Dirac(parameters)
        elif type == 'SAFE': #specific formulation for SAFE
            self.dist_change = parameters[0]
            self.HBN_factor = parameters[1]
        elif type == 'gamma':
            self.dist_change = ot.Gamma()
            self.dist_change.setParameter(ot.GammaMuSigma()(parameters))
    def plot_load_cdf(self):
        data = np.array(self.distribution.getParameter())
        x = np.split(data, 2)
        plt.plot(x[0], 1 - x[1])
        plt.yscale('log')
        plt.title('Probability of non-exceedence')
        plt.xlabel('Water level [m NAP]')
        plt.ylabel(r'$P_{non exceedence} (-/year)$')

#A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    def __init__(self, mechanism,computation_type,measure_year=0):
        #Initialize and make collection of MechanismReliability objects
        #mechanism, type, years are universal.
        # Measure_year is to indicate whether the reliability has to be recalculated or can be copied
        # (the latter is the case if a measure is taken later than the considered point in time)

        self.Reliability = {}

        for i in config.T:
            if measure_year > i:
                self.Reliability[str(i)] = MechanismReliability(mechanism, computation_type, copy_or_calculate='copy')
            else:
                self.Reliability[str(i)] = MechanismReliability(mechanism, computation_type)

    def generateInputfromDistributions(self, distributions, parameters = ['R', 'dR', 'S'], processes = ['dR']):
        processIDs = []
        for process in processes:
            processIDs.append(parameters.index(process))

        for i in self.Reliability:
            self.Reliability[i].Input.fill_distributions(distributions,np.int32(i),processIDs,parameters)

            pass

    def generateLCRProfile(self, load=False, mechanism='Overflow', method='FORM', trajectinfo=None, interpolate = 'False',conditionality = 'no'):
        # this function generates life-cycle reliability based on the years that have been calculated (so reliability in time)
        if load:
            [self.Reliability[i].calcReliability(self.Reliability[i].Input, load, mechanism=mechanism, method=method, year=float(i), TrajectInfo=trajectinfo) for i in self.Reliability.keys()]
        else:
            [self.Reliability[i].calcReliability(mechanism=mechanism, method=method,
                                                 year=float(i), TrajectInfo=trajectinfo) for i in
             self.Reliability.keys()]

        #NB: This could be extended with conditional failure probabilities

    def constructFragilityCurves(self,input,start = 5, step = 0.2):
        #Construct fragility curves for the entire collection
        for i in self.Reliability.keys():
            self.Reliability[i].constructFragilityCurve(self.Reliability[i].mechanism,input,year=i,start = start, step = step)

    def calcLifetimeProb(self, conditionality='no', period=None):
        #This script calculates the total probability over a certain period. It assumes independence of years.
        # This can be improved in the future to account for correlation.
        years = list(self.Reliability.keys())
        #set grid to range of calculations or defined period:
        tgrid = np.arange(np.int8(years[0]), np.int8(years[-1:]) + 1, 1) if period == None else np.arange(np.int8(years[0]), period, 1)
        t0 = []
        beta0 = []

        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)

        #calculate beta's per year, transform to pf, accumulate and then calculate beta for the period
        beta = np.interp(tgrid, t0, beta0)
        pfs = beta_to_pf(beta)
        pftot = 1 - np.cumprod(1 - pfs)
        self.beta_life = (np.max(tgrid), np.float(pf_to_beta(pftot[-1:])))

    def getProbinYear(self, year):
        #Interpolate a beta in a defined year from a collection of beta values
        t0 = []
        beta0 = []
        years = list(self.Reliability.keys())

        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)

        beta = np.interp(year, t0, beta0)
        return beta

    def drawLCR(self, yscale=None, type='beta', mechanism=None):
        #Draw the life cycle reliability. Default is beta but can be set to Pf
        t = []
        y = []

        for i in self.Reliability.keys():
            t.append(float(i)+config.t_0)
            if self.Reliability[i].type == 'Probabilistic':
                if self.Reliability[i].result.getClassName() == 'SimulationResult':
                    y.append(self.Reliability[i].result.getProbabilityEstimate()) if type == 'pf' else y.append(-ot.Normal().computeScalarQuantile(self.Reliability[i].result.getProbabilityEstimate()))
                else:
                    y.append(self.Reliability[i].result.getEventProbability()) if type == 'pf' else y.append(self.Reliability[i].result.getHasoferReliabilityIndex())
            else:
                y.append(self.Reliability[i].Pf) if type == 'pf' else y.append(self.Reliability[i].beta)

        plt.plot(t, y, label=mechanism)
        if yscale == 'log':
            plt.yscale(yscale)

        plt.xlabel('Time')
        plt.ylabel(r'$\beta$') if type != 'pf' else plt.ylabel(r'$P_f$')
        plt.title('Life-cycle reliability')

    def drawFC(self,yscale=None):
        #Drawa a fragility curve
        for j in self.Reliability.keys():
            wl = self.Reliability[j].wl
            pf = [self.Reliability[j].results[i].getProbabilityEstimate() for i in range(0, len(self.Reliability[j].results))]
            plt.plot(wl, pf, label=j)

        plt.legend()
        plt.ylabel('Pf|h[-/year]')
        plt.xlabel('h[m +NAP]')
        if yscale == 'log':
            plt.yscale('log')

    def drawAlphaBar(self,step = 5):
        import matplotlib.ticker as ticker

        alphas = np.array([])
        firstKey = list(self.Reliability.keys())[0]
        alphaDim = len(self.Reliability[firstKey].alpha_sq)
        alphas = np.concatenate([self.Reliability[i].alpha_sq for i in self.Reliability.keys()])
        alphas = np.reshape(alphas, (np.int(np.size(alphas) / alphaDim), alphaDim))
        variableNames = list(self.Reliability[firstKey].Input.input.getDescription())
        alphas = pd.DataFrame(alphas,columns = variableNames)
        ax = alphas.plot.bar(stacked=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        plt.ylim([0,1])
        plt.title(r'Influence coefficients $\alpha$ in time')
        plt.show()
class MechanismReliability:
    #This class contains evaluations of the reliability for a mechanism in a given year.
    def __init__(self, mechanism, type, copy_or_calculate='calculate'):
        #Initialize: set mechanism and type. These are the most important basic parameters
        self.mechanism = mechanism
        self.type = type
        self.copy_or_calculate = copy_or_calculate
        self.Input = MechanismInput(self.mechanism)
        if mechanism == 'Piping':
            self.gamma_schem_heave = 1 #1.05
            self.gamma_schem_upl = 1 #1.05
            self.gamma_schem_pip = 1 #1.05
        elif mechanism == 'StabilityInner':
            self.gamma_schem = 1 #1.1
        else:
            pass

    def __clearvalues__(self):
        #clear all values
        keys = self.__dict__.keys()
        for i in keys:
            # if i is not 'mechanism':
            if i != 'mechanism':
                setattr(self, i, None)

    def constructFragilityCurve(self, mechanism, input, start=5, step=0.2, method='MCS', splitPiping='no', year=0, lolim=10e-4, hilim=0.995):
        #Script to construct fragility curves from a mechanism model.
        if mechanism == 'Piping':
            #initialize the required lists
            marginals = []
            names = []
            if splitPiping == 'yes':
                Pheave = []
                Puplift = []
                Ppiping = []
                result_heave = []
                result_uplift = []
                result_piping = []
                wl_heave = []
                wl_uplift = []
                wl_piping = []
            else:
                Ptotal = []
                result_total = []
                wl_total = []

            # make list of all random variables:
            for i in input.input.keys():
                marginals.append(input.input[i])
                names.append(i)

            marginals.append(ot.Dirac(1.))
            names.append('h')

            if splitPiping == 'yes':
                result_heave, Pheave, wl_heave = IterativeFC_calculation(marginals, start, names, Mechanisms.zHeave, method, step, lolim, hilim)
                result_piping, Ppiping, wl_piping = IterativeFC_calculation(marginals, start, names, Mechanisms.zPiping, method, step, lolim, hilim)
                result_uplift, Puplift, wl_uplift = IterativeFC_calculation(marginals, start, names, Mechanisms.zUplift, method, step, lolim, hilim)
                self.h_cUplift = ot.Distribution(TableDist(np.array(wl_uplift), np.array(Puplift), extrap=True))
                self.resultsUplift = result_uplift
                self.wlUplift = wl_uplift
                self.h_cHeave = ot.Distribution(TableDist(np.array(wl_heave), np.array(Pheave), extrap=True))
                self.resultsHeave = result_heave
                self.wlHeave = wl_heave
                self.h_cPiping = ot.Distribution(TableDist(np.array(wl_piping), np.array(Ppiping), extrap=True))
                self.resultsPiping = result_piping
                self.wlPiping = wl_piping

            if splitPiping == 'no':
                result_total, Ptotal, wl_total = IterativeFC_calculation(marginals, start, names, Mechanisms.zPipingTotal, method, step, lolim, hilim)
                self.h_c = ot.Distribution(TableDist(np.array(wl_total), np.array(Ptotal), extrap=True))
                self.results = result_total
                self.wl = wl_total

        elif mechanism == 'Overflow':
            # make list of all random variables:
            marginals = []
            names = []

            for i in input.input.keys():
                #Adapt temporal process variables
                if i in input.temporals:
                    #Possibly this can be done using an OpenTurns random walk object
                    original = copy.deepcopy(input.input[i])
                    adapt_dist = TemporalProcess(original, year)
                    marginals.append(adapt_dist)
                    names.append(i)
                else:
                    marginals.append(input.input[i])
                    names.append(i)

            result = []
            marginals.append(ot.Dirac(1.)); names.append('h') #add the load
            result, P, wl = IterativeFC_calculation(marginals, start, names, Mechanisms.zOverflow, method, step, lolim, hilim)
            self.h_c = ot.Distribution(TableDist(np.array(wl), np.array(P), extrap=True))
            self.results = result
            self.wl = wl
        elif mechanism == 'StabilityInner':
            pass

        self.type = 'FragilityCurve'

    def calcReliability(self, strength = False, load = False, mechanism=None, method='FORM', year=0, TrajectInfo=None):
        #This routine calculates cross-sectional reliability indices based on different types of calculations.
        if self.type == 'DirectInput':
            pass
            #if input consists of 1 or 2 reliability values in time. Here we do an interpolation of those values to derive beta(year)
        if self.type == 'Simple':
            if mechanism == 'StabilityInner':
                if strength.input['SF_2025'].size != 0:
                    #Simple interpolation of two safety factors and translation to a value of beta at 'year'.
                    #In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
                    SFt = interpolate.interp1d([0, 50],np.concatenate((strength.input['SF_2025']/self.gamma_schem,
                                                                       strength.input['SF_2075']/self.gamma_schem)),fill_value='extrapolate')
                    SF = SFt(year)
                    modelfactor = 1.07 # Spencer, LiftVan = 1.06
                    beta = np.min([((SF/modelfactor)-0.41)/0.15, 8])
                elif strength.input['beta_2025'].size != 0:
                    #TODO check .gamma_schem
                    if np.size(strength.input['beta_2025']) == 0:
                        A=1
                    elif np.size(strength.input['beta_2075']) == 0:
                        A=1  #TODO uitzoeken waarom beta_2075 hier geen waarde heeft
                    betat = interpolate.interp1d([0, 50], np.concatenate((strength.input['beta_2025'] / self.gamma_schem,
                                                                    strength.input['beta_2075'] / self.gamma_schem)), fill_value='extrapolate')
                    beta = betat(year)
                else:
                    raise Exception('Warning: No inputvalues SF or Beta StabilityInner')
                # Check if there is an elimination measure present (diaphragm wall)
                if 'Elimination' in strength.input.keys():
                    if strength.input['Elimination'] == 'yes':
                        #Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                        self.Pf = beta_to_pf(beta) * strength.input['Pf_elim']  + \
                            strength.input['Pf_with_elim'] * (1-strength.input['Pf_elim'])
                        self.beta = pf_to_beta(self.Pf)
                    else:
                        raise ValueError('Warning: Elimination defined but not turned on')
                else:
                    self.beta = beta
                    self.Pf = beta_to_pf(self.beta)

            elif mechanism == 'Overflow': #specific for SAFE
                #climate change included, including a factor for HBN
                if hasattr(load,'dist_change'):
                    h_t = strength.input['h_crest'] - (strength.input['dhc(t)']+ (load.dist_change *load.HBN_factor)) * year
                else:
                    h_t = strength.input['h_crest'] - (strength.input['dhc(t)'] * year)

                self.beta, self.Pf = Mechanisms.OverflowSimple(h_t, strength.input['q_crest'], strength.input['h_c'], strength.input['q_c'], strength.input['beta'], mode='assessment')
            elif mechanism == 'Piping':
                pass

            self.alpha_sq = np.nan
            self.result = np.nan
        elif self.type == 'FragilityCurve':
            # Generic function for evaluating a fragility curve with water level and change in water level (optional)
            if hasattr(load, 'dist_change'):
                original = copy.deepcopy(load.dist_change)
                dist_change = TemporalProcess(original, year)
                #marginals = [self.Input.input['FC'], load, dist_change]
                P, beta = FragilityIntegration(self.Input.input['FC'], load, WaterLevelChange=dist_change)

                #result missing
                # dist = ot.ComposedDistribution(marginals)
                # dist.setDescription(['h_c', 'h', 'dh'])
                #TODO replace with FragilityIntegration. FragilityIntegration in ProbabilisticFunctions.py
                self.alpha_sq = np.nan
                self.result = np.nan
            else:
                marginals = [self.h_c, load.distribution]
                dist = ot.ComposedDistribution(marginals)
                dist.setDescription(['h_c', 'h'])
                result, P, beta, alfas_sq = run_prob_calc(ot.SymbolicFunction(['h_c', 'h'], ['h_c-h']), dist, method)
                self.result = result
                self.alpha_sq = alfas_sq
            self.Pf = P
            self.beta = beta

        elif self.type == 'Prob':
            #Probabilistic evaluation of a mechanism.
            if mechanism == 'Piping':
                zFunc = Mechanisms.zPipingTotal
            elif mechanism == 'Overflow':
                zFunc = Mechanisms.zOverflow
            elif mechanism == 'simpleLSF':
                zFunc = Mechanisms.simpleLSF
            else:
                raise ValueError('Unknown Z-function')

            if hasattr(self.Input,'char_vals'):
                start_vals = []
                for i in descr:
                    if i != 'h' and i != 'dh':
                        start_vals.append(strength.char_vals[i]) if i not in strength.temporals else start_vals.append(strength.char_vals[i] * year)
                start_vals = addLoadCharVals(start_vals, load)
            else:
                start_vals = self.Input.input.getMean()

            result, P, beta, alpha_sq = run_prob_calc(ot.PythonFunction(self.Input.input.getDimension(), 1, zFunc),
                                                      self.Input.input,
                                                      method, startpoint=start_vals)
            self.result = result
            self.Pf = P
            self.beta = beta
            self.alpha_sq = alpha_sq
        elif self.type == 'SemiProb':
            #semi probabilistic assessment, only available for piping
            if mechanism == 'Piping':
                if TrajectInfo == None: #Defaults, typical values for 16-3 and 16-4
                    TrajectInfo = {}; TrajectInfo['Pmax'] = 1. / 10000; TrajectInfo['omegaPiping'] = 0.24
                    TrajectInfo['bPiping'] = 300; TrajectInfo['aPiping'] = 0.9; TrajectInfo['TrajectLength'] = 20000
                # First calculate the SF without gamma for the three submechanisms
                # Piping:
                strength_new = copy.deepcopy(strength)

                for i in strength.temporals:
                    strength_new.input[i] = strength.input[i] * year
                # inputs = addLoadCharVals(strength_new.input, load=None, p_h=TrajectInfo['Pmax'], p_dh=0.5, year=year)
                # inputs['h'] = load.NormWaterLevel
                inputs = addLoadCharVals(strength_new.input, load=load, p_h=TrajectInfo['Pmax'], p_dh=0.5, year=year)

                Z, self.p_dh, self.p_dh_c = Mechanisms.zPiping(inputs, mode='SemiProb')
                self.gamma_pip = TrajectInfo['gammaPiping']
                # ProbabilisticFunctions.calc_gamma('Piping', TrajectInfo=TrajectInfo) #
                # Calculate needed safety factor

                if self.p_dh != 0 :
                    self.SF_p = (self.p_dh_c / (self.gamma_pip * self.gamma_schem_pip)) / self.p_dh
                else:
                    self.SF_p = 99
                self.assess_p = 'voldoende' if self.SF_p > 1 else 'onvoldoende'
                self.beta_cs_p = ProbabilisticFunctions.calc_beta_implicated('Piping', self.SF_p*self.gamma_pip,TrajectInfo=TrajectInfo)  #
                # Calculate the implicated beta_cs

                # Heave:
                Z, self.h_i, self.h_i_c = Mechanisms.zHeave(inputs,mode='SemiProb')
                self.gamma_h = TrajectInfo['gammaHeave'] #ProbabilisticFunctions.calc_gamma('Heave',TrajectInfo=TrajectInfo)  #
                # Calculate
                # needed safety factor

                self.SF_h = (self.h_i_c / (self.gamma_schem_heave * self.gamma_h)) / self.h_i
                self.assess_h = 'voldoende' if (self.h_i_c / (self.gamma_schem_heave * self.gamma_h)) / self.h_i > 1 else 'onvoldoende'
                self.beta_cs_h = ProbabilisticFunctions.calc_beta_implicated('Heave', (self.h_i_c/self.gamma_schem_heave) / self.h_i,TrajectInfo=TrajectInfo)  # Calculate the implicated beta_cs

                # Uplift
                Z, self.u_dh, self.u_dh_c = Mechanisms.zUplift(inputs,mode='SemiProb')
                self.gamma_u = TrajectInfo['gammaUplift'] #ProbabilisticFunctions.calc_gamma('Uplift',TrajectInfo=TrajectInfo)
                # Calculate
                # needed safety factor

                self.SF_u = (self.u_dh_c / (self.gamma_schem_upl * self.gamma_u)) / self.u_dh
                self.assess_u = 'voldoende' if (self.u_dh_c / (self.gamma_schem_upl * self.gamma_u)) / self.u_dh > 1 else 'onvoldoende'
                self.beta_cs_u = ProbabilisticFunctions.calc_beta_implicated('Uplift', (self.u_dh_c/self.gamma_schem_upl) / self.u_dh,TrajectInfo=TrajectInfo)  # Calculate the implicated beta_cs

                #Check if there is an elimination measure present (VZG or diaphragm wall)
                if 'Elimination' in strength.input.keys():
                    if strength.input['Elimination'] == 'yes':
                        #Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                        self.Pf = \
                            beta_to_pf(np.max([self.beta_cs_h,self.beta_cs_u,self.beta_cs_p])) * strength.input['Pf_elim']  + \
                            strength.input['Pf_with_elim'] * (1-strength.input['Pf_elim'])
                        self.beta = pf_to_beta(self.Pf)
                    else:
                        raise ValueError('Warning: Elimination defined but not turned on')
                else:
                    self.beta = np.min([np.max([self.beta_cs_h,self.beta_cs_u,self.beta_cs_p]),8])
                    self.Pf = beta_to_pf(self.beta)
                self.WLchar = copy.deepcopy(inputs['h']) #add water level as used in the assessment
                self.alpha_sq = np.nan
                self.result = np.nan
            else:
                pass

class MechanismInput:
    #Class for input of a mechanism
    def __init__(self,mechanism):
        self.mechanism = mechanism
    def fill_distributions(self,distributions,t,processIDs,parameters):

        dists = copy.deepcopy(distributions)
        self.temporals = []
        for j in processIDs:
            if t > 0:
                dists[j] = TemporalProcess(dists[j],t)
            else:
                dists[j] = ot.Dirac(0.0)
            self.temporals.append(parameters[j])
        self.input = ot.ComposedDistribution(dists)
        self.input.setDescription(parameters)

    #This routine reads  input from an input sheet
    def fill_mechanism(self, input, type= 'csv', sheet=None, calctype = 'Prob',mechanism=None):
        if type == 'csv':
            data = pd.read_csv(input, delimiter=',')
            if mechanism == 'Overflow':
                data = data.transpose()
            else:
                data = data.set_index('Name')
        elif type == 'xlsx':
            data = pd.read_excel(input,sheet_name=sheet)
            data = data.set_index('Name')

        self.input = {}
        self.temporals = []
        self.char_vals = {}
        for i in range(len(data)):
            # if (data.iloc[i].Name == 'FragilityCurve') and ~np.isnan(data.iloc[i].Value):
            if (data.index[i] == 'FragilityCurve'):
                # if ~np.isnan(data.iloc[i].Value): 
                if (~pd.isna(data.iloc[i].Value)== -1):
                    #csv inlezen en wegschrijven in self.input
                    FC = pd.read_csv(config.path.joinpath('FragilityCurve_STBI',data.iloc[i].Value), delimiter=';', header=0)

                    if np.min(np.diff(FC.beta))>0:  #beta values should be decreasing.
                        raise Exception('Fragility curve input should have decreasing betas. Filename: ' + data.iloc[i].Value)
                    if np.min(np.diff(FC.h))<0:  #h values should be increasing.
                        raise Exception('Fragility curve input should have increasing water levels. Filename: ' + data.iloc[i].Value )

                    A=np.argwhere(np.isnan(FC.values))
                    if A.size != 0:
                        FC.drop([A[0,0]])
                        raise Warning('NaN values in Fragility Curve from file ' + data.iloc[i].Value)
                    self.input['FC'] = FC

            else:
                x = data.iloc[i][:].values
                x = x.astype(np.float32)
                self.input[data.index[i]] = x[~np.isnan(x)]
                if data.index[i][-3:] == '(t)':
                    self.temporals.append(data.index[i])