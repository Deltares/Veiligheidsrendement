import openturns as ot
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

class Strategy:
    def __init__(self,type):
        self.Definition = StrategyDefinition(type)

class StrategyDefinition:
    def __init__(self,type):
        if type == 'dimensioninput':
            #here you can put in predefined dike reinforcement mdimensions
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
                    # self.Overflow[np.round(i,2)].constructFragilityCurves(input,start = input.input['h_c'].getParameter()[0],step = 0.1)
                    self.Overflow[np.round(i,2)].generateLCRProfile(params['load'],mechanism='Overflow',type='Prob',strength_params=input)
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