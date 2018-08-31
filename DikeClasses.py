import ProbabilisticFunctions
import Mechanisms
from HydraRing_scripts import DesignTableOpenTurns
import openturns as ot
import pandas as pd
import numpy as np
import copy
import inspect
from ProbabilisticFunctions import TableDist,run_DIRS, run_prob_calc, IterativeFC_calculation,TemporalProcess
from HelperFunctions import drawAlphaBarPlot
import matplotlib.pyplot as plt
from scipy.stats import norm
import openpyxl
#Class for the input of a mechanism. Function only available for piping for now
class LoadInput:
    #class to store load data
    def __init__(self):
        pass
    def set_fromDesignTable(self,filelocation,gridpoints=2000):
        #Load is given by exceedence probability-water level table from Hydra-Ring
        hist = DesignTableOpenTurns(filelocation,gridpoints=gridpoints)
        self.distribution = hist
    def set_annual_change(self,type='determinist',parameters = [0]):
        #set an annual change of the water level
        if type == 'determinist':
            self.dist_change = ot.Dirac(parameters)
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
        plt.ylabel(r'$P_{non exceedence} (-/year)$')class MechanismInput:
    #Class for input of a mechanism
    def __init__(self,mechanism):
        self.mechanism = mechanism
    #This routine fills a fragility curve with custom input parameters (can this be deleted and merged with the one below?)
    def fill_fragility_curve(self, hc_params, distribution = 'Normal'):
        if distribution == 'Normal':
            self.h_c = ot.Normal(hc_params[0],hc_params[1])

    #This routine reads probabilistic input from an input sheet
    def fill_prob_mechanism(self, input, sheet):
        data = pd.read_excel(input,sheet_name=sheet)
        data = data.set_index('Name')
        self.temporals = []; self.input = {}
        for i in range(0,len(data)):
            data.at[data.index[i],'Par2'] = data.at[data.index[i],'Par2']*data.at[data.index[i],'Par1'] if data.at[data.index[i],'variance type'] == 'var' else data.at[data.index[i],'Par2']
            if data.iloc[i]['Distribution'] == 'L':                 #Lognormal distribution
                self.input[data.index[i]] = ot.LogNormal()
                #With or without shift
                params = ot.LogNormalMuSigma()([float(data.iloc[i]['Par1']), float(data.iloc[i]['Par2']), float(data.iloc[i]['Par3'])]) if ~np.isnan(data.iloc[i]['Par3']) else ot.LogNormalMuSigma()([float(data.iloc[i]['Par1']), float(data.iloc[i]['Par2']), 0.0])
                self.input[data.index[i]].setParameter(params)
            elif data.iloc[i]['Distribution'] == 'N':               #Normal distribution
                self.input[data.index[i]] = ot.Normal(float(data.iloc[i]['Par1']),float(data.iloc[i]['Par2']))
            elif data.iloc[i]['Distribution'] == 'D':               #Determinist (Dirac) distribution
                self.input[data.index[i]] = ot.Dirac(float(data.iloc[i]['Par1']))
            elif data.iloc[i]['Distribution'] == 'G':               #Gamma distribution
                self.input[data.index[i]] = ot.Gamma()
                self.input[data.index[i]].setParameter(ot.GammaMuSigma()([float(data.iloc[i]['Par1']), float(data.iloc[i]['Par2']), float(data.iloc[i]['Par3'])]))
            else:
                raise TypeError('Input distribution type for parameter ' + data.index[i] + ' is not known')
            #Write value to temporals if required:
            self.temporals.append(data.index[i]) if data.iloc[i]['temporal'] == 'yes' else self.temporals

    def fill_piping_data(self, input,type):
        #Mechanism input from quickscan file
        self.inputtype = 'SemiProb'
        #Input needed for multiple submechanisms
        self.h           = input['Input']['Scenario 1']['j 0 '][1]
        self.h_exit      = input['Input']['Scenario 1']['hp = j 3'][1]
        self.gamma_w     = input['Input']['Scenario 1']['gw'][1]
        self.r_exit      = (input['Input']['Scenario 1']['j2'][1] - self.h_exit)/(input['Input']['Scenario 1']['j 0 '][1]-self.h_exit)
        self.d_cover     = input['Input']['Scenario 1']['d,pot'][1]
        self.gamma_schem = input['Input']['Scenario 1']['gb,u'][1]
        #Input parameter specifically for piping
        self.D           = input['Input']['Scenario 1']['D'][1]
        self.k           = input['Input']['Scenario 1']['kzand,pip'][1]
        self.L_voorland  = input['Input']['Scenario 1']['L1:pip'][1]
        self.L_dijk      = input['Input']['Scenario 1']['L2'][1]
        self.L_berm      = input['Input']['Scenario 1']['L3'][1]
        self.L           = self.L_berm + self.L_dijk + self.L_voorland
        self.d70         = input['Input']['Scenario 1']['d70'][1]
        self.d_cover_pip = input['Input']['Scenario 1']['d,pip'][1]
        self.m_Piping    = 1.0
        self.theta       = 37.
        #Input specific for heave
        self.scherm      = input['Results']['Scenario 1']['Heave']['kwelscherm aanwezig']
        #Input specific for uplift
        self.gamma_sat   = input['Input']['Scenario 1']['Gemiddeld volumegewicht:'][1]
        self.gamma_sat = 18 if self.gamma_sat == 0. else self.gamma_sat                 #Make sure the saturated weight is not 0

#Class describing safety assessments or descriptions (functions for piping, heave & uplift available)
class SectionReliability:
    def __init__(self):
        self
    def calcSectionReliability(self):
        self
        #This function should in the future calculate a section reliability based on all mechanism objects found in the class.

#A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    def __init__(self, mechanism,type,years):
        #Initialize and make collection of MechanismReliability objects
        self.Reliability = {}
        for i in years:
            self.Reliability[str(i)] = MechanismReliability(mechanism,type)

    def constructFragilityCurves(self,input,start = 5, step = 0.2):
        #Construct fragility curves for the entire collection
        for i in self.Reliability.keys():
            self.Reliability[i].constructFragilityCurve(self.Reliability[i].mechanism,input,year=i,start = start, step = step)
        pass

    def generateLCRProfile(self, load, mechanism = 'Overflow', method='FORM',type ='FragilityCurve',conditionality = 'no',strength_params=None):
        # this function generates life-cycle reliability based on the years that have been calculated
        if type == 'FragilityCurve':
            #For a fragility curve calculation
            [self.Reliability[i].calcFragilityCurve(load, mechanism=mechanism, method=method, year = float(i)) for i in self.Reliability.keys()]
        elif type == 'Simple':
            #For a user defined 'simple' calculation
            if mechanism == 'StabilityInner':
                [self.Reliability[i].calcSimple(strength_params, mechanism=mechanism, year = float(i)) for i in self.Reliability.keys()]

        #NB: This should be extended with conditional failure probabilities

    def calcLifetimeProb(self,conditionality='no',period=None):
        #This script calculates the total probability over a certain period. It assumes independence of years.
        # This can be improved in the future to account for correlation.

        years = list(self.Reliability.keys())
        #set grid to range of calculations or defined period:
        tgrid = np.arange(np.int8(years[0]),np.int8(years[-1:])+1,1) if period == None else np.arange(np.int8(years[0]),period,1)
        t0 = [];  beta0 = []
        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)
        #calculate beta's per year, transform to pf, accumulate and then calculate beta for the period
        beta = np.interp(tgrid, t0, beta0)
        pfs = norm.cdf(-beta)
        pftot = 1 - np.cumprod(1 - pfs)
        self.beta_life = (max(tgrid), np.float(norm.ppf(1 - pftot[-1:])))

    def getProbinYear(self,year):
        #Interpolate a beta in a defined year from a collection of beta values
        t0 = []; beta0 = []
        years = list(self.Reliability.keys())
        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)
        beta = np.interp(year,t0,beta0)
        return beta

    def parameterizeLCR(self):
        #Routine to parameterize the life-cycle reliability. Doesnt work properly for anything other than linear trends
        from scipy import stats
        t = []
        beta = []
        for i in self.Reliability.keys():
            t.append(int(i))
            beta.append(-ot.Normal().computeScalarQuantile(self.Reliability[i].Pf))
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, beta)
        tgrid = np.arange(0, 50, 1)
        betagrid = intercept + slope * tgrid
        plt.plot(t,beta,'or')
        plt.plot(tgrid, betagrid)
        print(r_value ** 2)

    def drawLCR(self,yscale=None,type='pf'):
        #Draw the life cycle reliability
        t = []; y = []
        for i in self.Reliability.keys():
            t.append(float(i))
            if self.Reliability[i].result.getClassName() == 'SimulationResult':
                y.append(self.Reliability[i].result.getProbabilityEstimate()) if type == 'pf' else y.append(-ot.Normal().computeScalarQuantile(self.Reliability[i].result.getProbabilityEstimate()))
            else:
                y.append(self.Reliability[i].result.getEventProbability()) if type == 'pf' else y.append(self.Reliability[i].result.getHasoferReliabilityIndex())
        plt.plot(t,y)
        if yscale=='log': plt.yscale(yscale)
        plt.xlabel('Time')
        plt.ylabel(r'$\beta$')
        plt.title('Life-cycle reliability')

    def drawFC(self,yscale=None):
        for j in self.Reliability.keys():
            wl = self.Reliability[j].wl
            pf = [self.Reliability[j].results[i].getProbabilityEstimate() for i in range(0, len(self.Reliability[j].results))]
            plt.plot(wl, pf, label=j)
        plt.legend()
        plt.ylabel('Pf|h[-/year]')
        plt.xlabel('h[m +NAP]')
        if yscale == 'log': plt.yscale('log')
#Continue refactoring and cleaning here! HEREHERE
class MechanismReliability:
    def __init__(self,mechanism,type):
        #Initialize: set mechanism and type. These are the most important basic parameters
        self.mechanism = mechanism
        self.type = type

    def __clearvalues__(self):
        #clear all values
        keys = self.__dict__.keys()
        for i in keys:
            if i is not 'mechanism':
                setattr(self,i,None)
    def constructFragilityCurve(self,mechanism,input,start = 5, step = 0.2,method='MCS',splitPiping = 'no',year = 0,lolim = 10e-4,hilim = 0.995):
        #Script to construct fragility curves from a mechanism model.
        if mechanism == 'Piping':
            #initialize the required lists
            marginals = []; names = [];
            if splitPiping == 'yes':
                Pheave = []; Puplift = []; Ppiping = []; result_heave = []; result_uplift = []; result_piping = []; wl_heave = []; wl_uplift = []; wl_piping = []
            else:
                Ptotal = []; result_total = []; wl_total = []

            # make list of all random variables:
            for i in input.input.keys():
                marginals.append(input.input[i])
                names.append(i)
            marginals.append(ot.Dirac(1.))
            names.append('h')

            if splitPiping == 'yes':
                result_heave,Pheave, wl_heave = IterativeFC_calculation(marginals, start, names, Mechanisms.zHeave, method, step, lolim, hilim)
                result_piping,Ppiping, wl_piping = IterativeFC_calculation(marginals, start, names, Mechanisms.zPiping, method, step, lolim, hilim)
                result_uplift,Puplift,wl_uplift = IterativeFC_calculation(marginals, start, names, Mechanisms.zUplift, method, step, lolim, hilim)
                self.h_cUplift = ot.Distribution(TableDist(np.array(wl_uplift), np.array(Puplift), extrap='on'))
                self.resultsUplift = result_uplift
                self.wlUplift = wl_uplift
                self.h_cHeave = ot.Distribution(TableDist(np.array(wl_heave), np.array(Pheave), extrap='on'))
                self.resultsHeave = result_heave
                self.wlHeave = wl_heave
                self.h_cPiping = ot.Distribution(TableDist(np.array(wl_piping), np.array(Ppiping), extrap='on'))
                self.resultsPiping = result_piping
                self.wlPiping = wl_piping
            if splitPiping == 'no':
                result_total,Ptotal, wl_total = IterativeFC_calculation(marginals, start, names, Mechanisms.zPipingTotal, method, step, lolim, hilim)
                self.h_c = ot.Distribution(TableDist(np.array(wl_total), np.array(Ptotal), extrap='on'))
                self.results = result_total
                self.wl = wl_total
        elif mechanism == 'Overflow':
            # make list of all random variables:
            marginals = []; names = []; wls = []
            for i in input.input.keys():
                #Adapt temporal process variables
                if i in input.temporals:
                    #Possibly this can be done using an OpenTurns random walk object
                    original = copy.deepcopy(input.input[i])
                    adapt_dist = TemporalProcess(original,year)
                    marginals.append(adapt_dist)
                    names.append(i)
                else:
                    marginals.append(input.input[i])
                    names.append(i)
            result = []
            marginals.append(ot.Dirac(1.)); names.append('h') #add the load
            result, P, wl = IterativeFC_calculation(marginals, start, names, Mechanisms.zOverflow, method, step, lolim, hilim)
            self.h_c = ot.Distribution(TableDist(np.array(wl), np.array(P), extrap='on'))
            self.results = result
            self.wl = wl
        elif mechanism == 'StabilityInner':
            pass
    def setFragilityCurve(self,dist):
        self.h_c = dist
    def calcSimple(self,strength,mechanism='StabilityInner', year=0):
        if mechanism == 'StabilityInner':
            self.beta = strength.input['beta'].getParameter()[0] - year * strength.input['dbeta'].getParameter()[0] + strength.input['berm_add'].getParameter()[0]*strength.input['beta_berm'].getParameter()[0]
            self.Pf = norm.cdf(-self.beta)
    def SemiProbabilistic(self,DikeSection, MechanismInput):
        #NOTE: this should be treated the same as the probabilistic and simple calculations
        if MechanismInput.mechanism == 'Piping':
            #First calculate the SF without gamma for the three submechanisms
            #Piping:
            Z, self.p_dh, self.p_dh_c = Mechanisms.LSF_sellmeijer(MechanismInput.h,MechanismInput.h_exit,MechanismInput.d_cover_pip,MechanismInput.L,MechanismInput.D,MechanismInput.d70,MechanismInput.k,MechanismInput.m_Piping)                   #Calculate hydraulic heads
            self.gamma_pip   = ProbabilisticFunctions.calc_gamma('Piping',DikeSection)                                         #Calculate needed safety factor
            #NB: Schematization factor IS NOT included here. Which is correct because a scenario approach is taken.
            self.SF_p = (self.p_dh_c/self.gamma_pip)/self.p_dh
            self.assess_p  = 'voldoende' if (self.p_dh_c/self.gamma_pip)/self.p_dh > 1 else 'onvoldoende'
            self.beta_cs_p = ProbabilisticFunctions.calc_beta_implicated('Piping',self.p_dh_c/self.p_dh,DikeSection)     #Calculate the implicated beta_cs

            #Heave:
            Z, self.h_i, self.h_i_c = Mechanisms.LSF_heave(MechanismInput.r_exit, MechanismInput.h, MechanismInput.h_exit, MechanismInput.d_cover_pip, MechanismInput.scherm)                                  #Calculate hydraulic heads
            self.gamma_h   = ProbabilisticFunctions.calc_gamma('Heave',DikeSection)                                            #Calculate needed safety factor
            # Check if it is OK, NB: Schematization factor IS included here
            self.SF_h = (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i
            self.assess_h  = 'voldoende' if (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i > 1 else 'onvoldoende'
            self.beta_cs_h = ProbabilisticFunctions.calc_beta_implicated('Heave',self.h_i_c/self.h_i,DikeSection)                 #Calculate the implicated beta_cs
            #Uplift
            Z, self.u_dh, self.u_dh_c = Mechanisms.LSF_uplift(MechanismInput.r_exit, MechanismInput.h, MechanismInput.h_exit, MechanismInput.d_cover_pip, MechanismInput.gamma_sat)                                  #Calculate hydraulic heads
            self.gamma_u   = ProbabilisticFunctions.calc_gamma('Uplift',DikeSection)                                            #Calculate needed safety factor
            #NB: Schematization factor IS included here
            self.SF_u = (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh
            self.assess_u  = 'voldoende' if (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh > 1 else 'onvoldoende'
            self.beta_cs_u = ProbabilisticFunctions.calc_beta_implicated('Uplift',self.u_dh_c/self.u_dh,DikeSection)                 #Calculate the implicated beta_cs
        else:
            pass
    def calcFragilityCurve(self, load, method='FORM', mechanism=None, year=0):
        #Generic function for evaluating a fragility curve with water level and change in water level (optional)
        if hasattr(load,'dist_change'):
            original = copy.deepcopy(load.dist_change)
            dist_change = TemporalProcess(original, year)
            marginals = [self.h_c, load.distribution, dist_change]
            dist = ot.ComposedDistribution(marginals)
            dist.setDescription(['h_c', 'h', 'dh'])
            result,P,beta,alfas_sq = run_prob_calc(ot.SymbolicFunction(['h_c', 'h', 'dh'], ['h_c-(h+dh)']),dist,method)
        else:
            marginals = [self.h_c, load.distribution]
            dist = ot.ComposedDistribution(marginals)
            dist.setDescription(['h_c', 'h'])
            result,P,beta,alfas = run_prob_calc(ot.SymbolicFunction(['h_c', 'h'], ['h_c-h']),dist,method)
        self.result = result
        self.Pf = P
        self.beta = beta
        self.alpha_sq = alfas_sq


#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.TrajectInfo = {}
        self.Reliability = SectionReliability()

        #Basic traject info
        if traject == '16-4':
            self.TrajectInfo['TrajectLength'] = 19480
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
            self.name = name[0:5] + '0' + name[5:] if len(name) == 8 else name  # Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
        elif traject == '16-3':
            self.TrajectInfo['TrajectLength'] = 19899
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
            self.name = name[0:5] + '0' + name[5:] if len(name) == 8 else name  # Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
            #NB: klopt a hier?????!!!!

    def fill_from_dict(self, dict):
        #Fill the data from a dictionary that has been generated using readQuickScan.py (i.e. from the pickle files)
        #This concerns data from the piping quickscan

        #First the general info (to be added: traject info, norm etc)
        self.start = dict['General']['Traject start']
        self.end   = dict['General']['Traject end']
        self.CS    = dict['General']['Cross section']
        self.MHW   = dict['Input']['Scenario 1']['j 0 ']        #TO DO: add a loop over scenarios
        self.Reliability.Piping = MechanismReliability('Piping','SemiProb')
        self.Reliability.Piping.Input = MechanismInput('Piping')
        self.Reliability.Piping.Input.fill_piping_data(dict,'SemiProb')  #fill the data for the piping assessment

    def readGeneralInfo(self,path,sheet):
        #Read general data from sheet in standardized xlsx file
        data = pd.read_excel(path,sheet_name=sheet)
        data = data.set_index('Name')
        #We read all the data and make an exception for the Initial Geometry where we have to translate the string of points to a list of points
        for i in range(0, len(data)):
            if data.index[i] == 'InitialGeometry':
                points_in = data.loc[data.index[i]]
                geometry = points_in[0].split(';')
                for j in range(0, len(geometry)):
                    geometry[j] = list(map(float, geometry[j].split(',')))
                setattr(self, data.index[i], geometry)
            else:
                setattr(self, data.index[i], data.loc[data.index[i]][0])
        print()

    def doAssessment(self, mechanism, type):
        #Routine to do an assessment: change this so it fits with 'Prob' and 'Simple' calculations
        if mechanism == 'Piping':
            # self.Reliability.Piping= MechanismReliability(mechanism, type)
            self.Reliability.Piping.SemiProbabilistic(self,self.Reliability.Piping.Input)
        else:
            print('Mechanism not known')

class Strategy:
    def __init__(self,type):
        self.Definition = StrategyDefinition(type)

class StrategyDefinition:
    def __init__(self,type):
        if type == 'dimensioninput':
            #here you can put in predefined dike reinforcement dimensions
            StrategyDefinition.type = 'dimensioninput'
        elif type == 'optimizationpersection':
            #here we optimize so that a section meets the standard
            StrategyDefinition.type = 'optimizationpersection'
        elif type == 'optimizationpersegment':
            pass
            #here we optimize so that the segment meets the standard

    def readinput(self,inputfile,sheet):
        if self.type == 'optimizationpersection':
            data = pd.read_excel(inputfile, sheet_name=sheet)
            self.measures = {}
            self.measures[data.index[0][0]] = {}
            for i in range(0, len(data)):
                self.measures[data.index[0][0]][data.index[i][1]] = data.loc[data.index[i]][0]
        else:
            pass

class InitialSituation:
    #Put the initial situation (i.e. no measures) in the strategy object
    def __init__(self,SectionReliabilityObject):
        self.SectionReliability = SectionReliabilityObject

class StrategyCalculation:
    def __init__(self,Definition,InitialSituation):
        ## Derive dh - beta0 relation for all available measures
        measurenames = list(Definition.measures.keys())
        self.Measures = {}
        for i in measurenames:
            params = {}
            if i == 'Soil reinforcement':
                #Define the ranges that are possible for soil reinforcement
                dcrest_min = Definition.measures[i]['dcrest_min']; dcrest_max = Definition.measures[i]['dcrest_max']
                params['outwardrange'] = np.linspace(0,Definition.measures[i]['max_outward'],10) if Definition.measures[i]['max_outward'] >0 else np.nan
                params['inwardrange'] = np.linspace(0, Definition.measures[i]['max_inward'], 10) if Definition.measures[i]['max_inward'] > 0 else np.nan
                params['crestrange']   = np.linspace(dcrest_min,dcrest_max,6)
                #Initial situation:
                params['initial_overflow'] = InitialSituation.SectionReliability.Overflow.Input
                params['initial_stabilityinner'] = InitialSituation.SectionReliability.StabilityInner.Input
                params['load']         = InitialSituation.SectionReliability.Load
            #Evaluate measures for different mechanisms
            self.Measures[i] = {}
            self.Measures[i]['Overflow'] = MeasureEvaluation('Soil reinforcement','Overflow',params)
            self.Measures[i]['StabilityInner'] = MeasureEvaluation('Soil reinforcement','StabilityInner',params)

class MeasureEvaluation:
    #Class for the evaluation of a measure
    def __init__(self,type,mechanism,params):
        years = [1,10,20,30,40,50,70,100]
        #Evaluation of soil reinforcement
        if type == 'Soil reinforcement':
            if mechanism == 'Overflow':
                self.Overflow = {}
                input0 = copy.deepcopy(params['initial_overflow'])
                for i in params['crestrange']:
                    input = copy.deepcopy(input0)
                    if isinstance(input.input['h_c'],ot.Normal):
                        input.input['h_c'] = ot.Normal(input.input['h_c'].getParameter()[0]+i,input.input['h_c'].getParameter()[1])
                    else:
                        raise ValueError('Unknown distribution type for h_c')
                    self.Overflow[np.round(i,2)] = MechanismReliabilityCollection('Overflow','Prob',years)
                    self.Overflow[np.round(i,2)].constructFragilityCurves(input,start = input.input['h_c'].getParameter()[0],step = 0.1)
                    self.Overflow[np.round(i,2)].generateLCRProfile(params['load'],mechanism='Overflow')
            if mechanism == 'StabilityInner':
                self.StabilityInner = {}
                input0 = copy.deepcopy(params['initial_stabilityinner'])
                #Make a tuple of max inner and outer berm (with the simple model they have the same effect)
                if np.isnan(params['inwardrange']):
                    variants = [(round(x,1), round(y,1)) for x in [0] for y in params['outwardrange']]
                elif np.isnan(params['outwardrange']):
                    variants = [(round(x,1), round(y,1)) for x in params['inwardrange'] for y in [0]]
                else:
                    variants = [(round(x,1), round(y,1)) for x in params['inwardrange'] for y in params['outwardrange']]
                # very simple mechanism model for now
                for i in variants:
                    input = copy.deepcopy(input0)
                    input.input['berm_add'] = ot.Dirac(i[0]+i[1])
                    self.StabilityInner[i] = MechanismReliabilityCollection('StabilityInner','Simple',years)
                    self.StabilityInner[i].generateLCRProfile(params['load'],mechanism='StabilityInner',type='Simple',strength_params=input)

    def deriveBetaRelations(self,mechanism,measure_type, period = None):
        #Script to derive beta relations for increments of a reinforcement type
        if measure_type == 'Soil reinforcement':
            if mechanism == 'Overflow':
                dh = []; beta_0 = []; beta_min = []; beta_life = []
                if period == None: period = int8(self.Overflow.keys()[-1:])
                for i in self.Overflow.keys():
                    dh.append(np.float(i))
                    beta_0.append(self.Overflow[i].Reliability['1'].beta)
                    beta_min.append(self.Overflow[i].getProbinYear(period))
                    self.Overflow[i].calcLifetimeProb(period=period)
                    beta_life.append(self.Overflow[i].beta_life[1])
                self.betadH = {}
                self.betadH = pd.DataFrame({'dh': dh, 'beta_0': beta_0,'beta_life': beta_life, 'beta_min': beta_min})
                self.betadH.set_index('dh')
            elif mechanism == 'StabilityInner':
                dBermIn = []
                dBermOut = []
                dBermInOut = []
                beta_0 = []
                beta_min = []
                beta_life = []
                if period == None:
                    period = int8(self.StabilityInner.keys()[-1:])
                for i in self.StabilityInner.keys():
                    dBermIn.append(np.round(i[0],1))
                    dBermOut.append(np.round(i[1],1))
                    dBermInOut.append((np.round(i[0],1),np.round(i[1],1)))
                    beta_0.append(self.StabilityInner[i].Reliability['1'].beta)
                    beta_min.append(self.StabilityInner[i].getProbinYear(period))
                    self.StabilityInner[i].calcLifetimeProb(period=period)
                    beta_life.append(self.StabilityInner[i].beta_life[1])
                self.betadBerm = {}
                self.betadBerm = pd.DataFrame({'dBerm (Inward/Outward)': dBermInOut, 'beta_0': beta_0,'beta_life': beta_life, 'beta_min': beta_min})
                self.betadBerm.set_index('dBerm (Inward/Outward)')
            elif mechanism == 'Piping':
                pass
        else:
            pass

    def plotMeasureReliabilityinTime(self):
        #Plot the reliability of different measure steps in time
        mechanism = inspect.getmembers(self)[0][0]
        if mechanism == 'StabilityInner':
            for i in self.StabilityInner.keys():
                t = []
                beta = []
                for j in self.StabilityInner[i].Reliability.keys():
                    t.append(np.float(j))
                    beta.append(self.StabilityInner[i].Reliability[j].beta)
                plt.plot(t,beta, label=str(i))
            plt.title('Reliability in time for different combinations of inner and outer berm widening')
        elif mechanism == 'Overflow':
            for i in self.Overflow.keys():
                t = []
                beta = []
                for j in self.Overflow[i].Reliability.keys():
                    t.append(np.float(j))
                    beta.append(self.Overflow[i].Reliability[j].beta)
                plt.plot(t,beta, label=str(i))
            plt.title('Reliability in time for different increments of crest level increase')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel(r'$\beta$')
        plt.show()
