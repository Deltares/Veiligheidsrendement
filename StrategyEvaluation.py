import openturns as ot
import time
from collections import OrderedDict
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from Mechanisms import OverflowSimple
from scipy.stats import norm
from scipy.interpolate import interp1d
import math
from shapely.geometry import Polygon
from DikeClasses import MechanismReliabilityCollection, SectionReliability
import cProfile
import re
class Solutions:
    def __init__(self, DikeSectionObject):
        self.SectionName = DikeSectionObject.name
        self.Length = DikeSectionObject.Length
        self.InitialGeometry = DikeSectionObject.InitialGeometry

    def fillSolutions(self,excelsheet):
        data = pd.read_excel(excelsheet,'Measures')
        self.Measures = {}
        partials = []
        for i in data.index:
            self.Measures[i] = Measure(data.loc[i])
        self.MeasureTable = pd.DataFrame(columns=['ID', 'Name'])
        for i in range(0,len(self.Measures)):
            self.MeasureTable.loc[i] = [str(self.Measures[i].parameters['ID']), self.Measures[i].parameters['Name']]


    def evaluateSolutions(self,DikeSection,TrajectInfo,trange = [0,24,25,50,75,100], geometry_plot='off',plot_dir = None):
        self.trange = trange
        removal = []
        for i in self.Measures:
            if self.Measures[i].parameters['available'] == 1:
                self.Measures[i].evaluateMeasure(i,DikeSection, TrajectInfo, trange = trange, geometry_plot=geometry_plot, plot_dir = plot_dir)
            else:
                removal.append(i)
        if len(removal) > 0 :
            for i in removal:
                self.Measures.pop(i)
        print()
    def SolutionstoDataFrame(self,filtering='off'):
        mechanisms = list(self.Measures[list(self.Measures.keys())[0]].measures[0]['Reliability'].Mechanisms.keys()); mechanisms.append('Section')
        years = self.trange
        cols_r = pd.MultiIndex.from_product([mechanisms,years],names =['base','year'])
        reliability = pd.DataFrame(columns = cols_r)
        cols_m = pd.Index(['ID','type','class','year','params','cost'],name='base')
        measure = pd.DataFrame(columns = cols_m)
        # data = pd.DataFrame(columns = cols)
        inputs_m = [];
        inputs_r = [];
        for i in list(self.Measures.keys()):

            if isinstance(self.Measures[i].measures, list):
                typee = self.Measures[i].parameters['Type']
                for j in range(0, len(self.Measures[i].measures)):
                    measure_in = []; reliability_in = []
                    if typee == 'Soil reinforcement': designvars = str((self.Measures[i].measures[j]['dcrest'], self.Measures[i].measures[j]['dberm']))
                    cost = self.Measures[i].measures[j]['Cost']
                    measure_in.append(str(self.Measures[i].parameters['ID'])); measure_in.append(typee); measure_in.append(self.Measures[i].parameters['Class']);
                    measure_in.append(self.Measures[i].parameters['year'])
                    measure_in.append(designvars); measure_in.append(cost)

                    betas = self.Measures[i].measures[j]['Reliability'].SectionReliability
                    for ij in mechanisms:
                        for ijk in betas.loc[ij].values:
                            reliability_in.append(ijk)

                    inputs_m.append(measure_in); inputs_r.append(reliability_in)

            elif isinstance(self.Measures[i].measures, dict):
                # inputs_m = []; inputs_r = [];
                ID = str(self.Measures[i].parameters['ID'])
                typee = self.Measures[i].parameters['Type']
                if typee == 'Vertical Geotextile': designvars = self.Measures[i].measures['VZG']
                if typee == 'Diaphragm Wall': designvars = self.Measures[i].measures['DiaphragmWall']
                classe = self.Measures[i].parameters['Class']
                yeare  = self.Measures[i].parameters['year']
                cost = self.Measures[i].measures['Cost']
                inputs_m.append([ID, typee, classe, yeare, designvars,cost]);
                betas = self.Measures[i].measures['Reliability'].SectionReliability
                beta = []
                for ij in mechanisms:
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)
        reliability = reliability.append(pd.DataFrame(inputs_r, columns=cols_r))
        measure = measure.append(pd.DataFrame(inputs_m, columns=cols_m))
        self.MeasureData = measure.join(reliability,how='inner')
        #fix multiindex
        index = []
        for i in self.MeasureData.columns:
            index.append(i) if isinstance(i,tuple) else index.append((i,''))
        self.MeasureData.columns = pd.MultiIndex.from_tuples(index)
        if filtering == 'on':
            pass
    def plotBetaTimeEuro(self, measures='undefined',mechanism='Section',beta_ind = 'beta0',sectionname='Unknown',beta_req=None):
        #measures is a list of measures that need to be plotted
        if measures == 'undefined': measures = list(self.Measures.keys())
        #mechanism can be used to select a single or all 'Section' mechanisms
        #beta can be used to use a criterion for selecting the 'best' designs, such as the beta at 't0'
        cols = ['type','parameters','Cost']
        [cols.append('beta'+str(i)) for i in self.trange]
        data = pd.DataFrame(columns = cols)
        num_plots = 5
        colors = sns.color_palette('hls', n_colors=num_plots)
        # colors = plt.cm.get_cmap(name=plt.cm.hsv, lut=num_plots)
        color=0
        for i in np.unique(self.MeasureData['ID'].values):
            if isinstance(self.Measures[int(i)-1].measures, list):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID']==i])
                # inputs = []; type = self.Measures[i].parameters['Type']
                # for j in range(0, len(self.Measures[i].measures)):
                #     inputvals = []
                #     if type == 'Soil reinforcement': designvars = str((self.Measures[i].measures[j]['dcrest'], self.Measures[i].measures[j]['dberm']))
                #     betas = list(self.Measures[i].measures[j]['Reliability'].SectionReliability.loc[mechanism])
                #     cost = self.Measures[i].measures[j]['Cost']
                #     inputvals.append(type); inputvals.append(designvars); inputvals.append(cost)
                #     for ij in range(0, len(betas)): inputvals.append(betas[ij])
                #     inputs.append(inputvals)
                # data = data.append(pd.DataFrame(inputs, columns=cols))
                # x = data.loc[data['type'] == 'Soil reinforcement']
                y = copy.deepcopy(data)
                x = data.sort_values(by=['cost'])

                steps = 20
                cost_grid = np.linspace(np.min(x['cost']), np.max(x['cost']), steps)
                envelope_beta = []
                envelope_costs = []
                indices = []
                betamax = 0
                for j in range(0, len(cost_grid) - 1):
                    values = x.loc[(x['cost'] >= (cost_grid[j])) & (x['cost'] <= (cost_grid[j + 1]))][(mechanism,beta_ind)]
                    if len(list(values)) > 0:
                        idd = values.idxmax()
                        if betamax < np.max(list(values)):
                            betamax = np.max(list(values))
                            indices.append(idd)
                            if isinstance(x['cost'].loc[idd],pd.Series): envelope_costs.append(x['cost'].loc[idd].values[0])
                            if not isinstance(x['cost'].loc[idd], pd.Series): envelope_costs.append(x['cost'].loc[idd])
                            envelope_beta.append(betamax)
                if self.Measures[np.int(i)-1].parameters['Name'][-4:] != '2045':
                    plt.plot(envelope_costs, envelope_beta, color=colors[color], linestyle='-')
                    # [plt.text(y['Cost'].loc[ij], y[beta_ind].loc[i], y['parameters'].loc[ij],fontsize='x-small') for ij in indices]

                    plt.plot(y['cost'], y[(mechanism,beta_ind)], label = self.Measures[np.int(i)-1].parameters['Name'],
                             marker='o',markersize=6, color=colors[color],markerfacecolor=colors[color],
                             markeredgecolor=colors[color], linestyle='',alpha=1)

                    color += 1
            elif isinstance(self.Measures[np.int(i)-1].measures, dict):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID']==i])
                #
                # inputs = []; type = self.Measures[np.int(i)].parameters['Type']
                # if type == 'Vertical Geotextile': designvars = self.Measures[np.int(i)].measures['VZG']
                # if type == 'Diaphragm Wall': designvars = self.Measures[np.int(i)].measures['DiaphragmWall']
                # betas = list(self.Measures[np.int(i)].measures['Reliability'].SectionReliability.loc[mechanism])
                # cost = self.Measures[np.int(i)].measures['Cost']
                # inputs.append(type); inputs.append(designvars); inputs.append(cost);
                # for ij in range(0, len(betas)): inputs.append(betas[ij])
                # data = data.append(pd.DataFrame([inputs], columns=cols))
                plt.plot(data['cost'], data[(mechanism,beta_ind)], label = self.Measures[np.int(i)-1].parameters['Name'],
                         marker='d',markersize=10,markerfacecolor=colors[color],markeredgecolor=colors[color],linestyle='')
                color += 1
        axes = plt.gca()
        plt.plot([0, axes.get_xlim()[1]], [beta_req, beta_req], 'k--', label='Norm')
        plt.xlabel('Cost');
        plt.ylabel(r'$\beta_{' + str(beta_ind+2025) + '}$')
        plt.title('Cost-beta relation for ' + mechanism + ' at ' + sectionname)
        plt.legend(loc='best')

class Measure:
    def __init__(self,inputs):
        self.parameters = {}
        for i in range(0,len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]
    def evaluateMeasure(self,name,DikeSection,TrajectInfo, trange=None, geometry_plot='off', t0 = 2025, plot_dir = None):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        SFincrease = 0.2
        if geometry_plot=='on': plt.figure(1000)
        if type == 'Soil reinforcement':
            crest_step = 0.5; berm_step = 10
            # crest_step = 0.25; berm_step = 10

            crestrange = np.linspace(self.parameters['dcrest_min'], self.parameters['dcrest_max'],
                                     1+(self.parameters['dcrest_max']-self.parameters['dcrest_min'])/crest_step)
            if self.parameters['Direction'] == 'outward':
                bermrange = np.linspace(0., self.parameters['max_outward'], 1+(self.parameters['max_outward']/berm_step))
            elif self.parameters['Direction'] == 'inward':
                bermrange = np.linspace(0., self.parameters['max_inward'], 1+(self.parameters['max_inward']/berm_step))
            measures = [(x,y) for x in crestrange for y in bermrange]
            slope_inner = 4; slope_outer = 3;
            self.measures = []
            if self.parameters['StabilityScreen'] == 'yes':
                self.parameters['Depth'] = DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input['d_cover'] + 1.

            for j in measures:
                self.measures.append({})
                self.measures[-1]['dcrest'] =j[0]
                self.measures[-1]['dberm'] = j[1]
                self.measures[-1]['Geometry'], area_difference = DetermineNewGeometry(j,slope_inner,slope_outer,self.parameters['Direction'],DikeSection.InitialGeometry,geometry_plot=geometry_plot, plot_dir = plot_dir)
                self.measures[-1]['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, reinf_pars = j, housing = DikeSection.houses, area_difference= area_difference)
                self.measures[-1]['Reliability'] = SectionReliability()
                self.measures[-1]['Reliability'].Mechanisms = {}
                for i in mechanisms:
                    calc_type = DikeSection.MechanismData[i][1]
                    if trange == None:
                        trange = [int(i) for i in DikeSection.Reliability.Mechanisms[i].Reliability.keys()]
                    self.measures[-1]['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i,calc_type,years = trange,measure_year = self.parameters['year'] )
                    # self.measures[-1]['Reliability'].Mechanisms[i].Input = {}
                    for ij in self.measures[-1]['Reliability'].Mechanisms[i].Reliability.keys():
                        self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                        #Adapt inputs
                        if float(ij) >= self.parameters['year']: #year of finishing improvement should be given.
                            if i == 'Overflow':
                                self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['h_crest'] = \
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['h_crest'] + self.measures[-1]['dcrest']
                            elif i == 'StabilityInner':
                                #NOTE: we do not account for the slope reduction. This should be implemented for outward reinforcements.
                                if self.parameters['Direction'] == 'inward':
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                                    if self.parameters['StabilityScreen'] == 'yes':
                                        self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] += SFincrease
                                        self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] += SFincrease

                                elif self.parameters['Direction'] == 'outward':
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                            elif i == 'Piping':
                                self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lvoor'] = \
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lvoor'] + self.measures[-1]['dberm']
                                self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lachter'] = \
                                    np.max([0.,self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lachter'] - self.measures[-1]['dberm']])
                        # self.measures[-1]['Reliability'].Mechanisms[i].Reliability = {}
                    self.measures[-1]['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
                self.measures[-1]['Reliability'].calcSectionReliability(TrajectInfo,DikeSection.Length)
                    #Calc betas!
        elif type == 'Vertical Geotextile':
            #No influence on overflow and stability
            #Only 1 parameterized version with a lifetime of 50 years
            if trange == None:
                trange = [int(i) for i in DikeSection.Reliability.Mechanisms[i].Reliability.keys()]
            self.measures = {}
            self.measures['VZG'] = 'yes'
            self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
            self.measures['Reliability'] = SectionReliability()
            self.measures['Reliability'].Mechanisms = {}

            for i in mechanisms:
                calc_type = DikeSection.MechanismData[i][1]
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=trange)
                for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(
                        DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                    if i == 'Overflow' or i == 'StabilityInner' or (i == 'Piping' and int(ij) < self.parameters['year']): #Copy results
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij] = copy.deepcopy(
                            DikeSection.Reliability.Mechanisms[i].Reliability[ij])
                    elif i == 'Piping' and int(ij) >= self.parameters['year']:
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Elimination'] = 'yes'
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_elim'] = self.parameters['P_solution']
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_with_elim'] = self.parameters['Pf_solution']
                self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
            self.measures['Reliability'].calcSectionReliability(TrajectInfo,DikeSection.Length)
        elif type == 'Diaphragm Wall':
            #StabilityInner and Piping reduced to 0, height is ok for overflow until 2125 (free of charge, also if there is a large height deficit).
            # It is assumed that the diaphragm wall is extendable after that.
            #Only 1 parameterized version with a lifetime of 100 years
            self.measures = {}
            self.measures['DiaphragmWall'] = 'yes'
            self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
            self.measures['Reliability'] = SectionReliability()
            self.measures['Reliability'].Mechanisms = {}
            for i in mechanisms:
                calc_type = DikeSection.MechanismData[i][1]
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=trange)
                for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                    if float(ij) >= self.parameters['year']:
                        if i == 'Overflow':
                            Pt = TrajectInfo['Pmax']*TrajectInfo['omegaOverflow']
                            if hasattr(DikeSection,'HBNRise_factor'):
                                hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange = DikeSection.HBNRise_factor * DikeSection.YearlyWLRise, mechanism='Overflow')
                            else:
                                hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange=None, mechanism='Overflow')
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['h_crest'] = hc
                        elif i == 'StabilityInner' or i == 'Piping':
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Elimination'] = 'yes'
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_elim'] = self.parameters['P_solution']
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_with_elim'] = self.parameters['Pf_solution']
                self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
            self.measures['Reliability'].calcSectionReliability(TrajectInfo,DikeSection.Length)
        elif type == 'Stability Screen':
            self.measures = {}
            self.measures['Stability Screen'] = 'yes'
            SFincrease = 0.2
            self.parameters['Depth'] = DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input['d_cover'] + 1.
            self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
            self.measures['Reliability'] = SectionReliability()
            self.measures['Reliability'].Mechanisms = {}
            for i in mechanisms:
                calc_type = DikeSection.MechanismData[i][1]
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=trange)
                for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                    if i == 'Overflow' or i == 'Piping': #Copy results
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij] = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij])
                        pass #no influence
                    elif i == 'StabilityInner':
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                        if int(ij)>=self.parameters['year']:
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] += SFincrease
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] += SFincrease
                self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
            self.measures['Reliability'].calcSectionReliability(TrajectInfo,DikeSection.Length)
class Strategy:
    def __init__(self,type,r=0.03):
        self.type = type        #OI or CB
        self.r = r
    def combine(self, traject, solutions, filtering ='off',OI_horizon=50,OI_year = 0):
        self.options = {}

        # cols = ['name'] + list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)
        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # measures at t=0 (2025) and t=20 (2045)
        for i in range(0, len(traject.Sections)):
            sec = traject.Sections[i]
            # Step 1: combine measures with partial measures
            combinables = solutions[traject.Sections[i].name].MeasureData.loc[solutions[traject.Sections[i].name].MeasureData['class'] == 'combinable']
            if self.type == 'OI' and isinstance(OI_year,int):
                combinables = combinables.loc[solutions[traject.Sections[i].name].MeasureData['year'] == OI_year]
            partials = solutions[traject.Sections[i].name].MeasureData.loc[solutions[traject.Sections[i].name].MeasureData['class'] == 'partial']
            if self.type == 'OI' and isinstance(OI_year,int):
                partials = partials.loc[solutions[traject.Sections[i].name].MeasureData['year'] == OI_year]
            combinedmeasures = MeasureCombinations(combinables, partials,solutions[traject.Sections[i].name])
            # make sure combinable, mechanism and year are in the MeasureData dataframe
            # make a strategies dataframe where all combinable measures are combined with partial measures for each timestep
            #if there is a measureid that is not known yet, add it o the measure table
            # i = 3
            existingIDs = solutions[traject.Sections[i].name].MeasureTable['ID'].values
            IDs = np.unique(combinedmeasures['ID'].values)
            if len(IDs) >0:
                for ij in IDs:
                    name = solutions[traject.Sections[i].name].MeasureTable.loc[solutions[traject.Sections[i].name].MeasureTable['ID'] == ij[0]]['Name'].values + \
                           '+' + solutions[traject.Sections[i].name].MeasureTable.loc[solutions[traject.Sections[i].name].MeasureTable['ID'] == ij[1]]['Name'].values
                    solutions[traject.Sections[i].name].MeasureTable.loc[len(solutions[traject.Sections[i].name].MeasureTable) + 1] = ['+'.join(ij), str(name[0])]

            StrategyData = copy.deepcopy(solutions[traject.Sections[i].name].MeasureData)
            if self.type == 'OI' and isinstance(OI_year,int):
                StrategyData = StrategyData.loc[StrategyData['year']==OI_year]

            StrategyData = StrategyData.append(combinedmeasures)

            if filtering == 'on':
                StrategyData = copy.deepcopy(StrategyData)
                StrategyData = StrategyData.reset_index(drop=True)
                LCC = calcTC(StrategyData)
                ind = np.argsort(LCC)
                LCC_sort = LCC[ind]
                StrategyData = StrategyData.iloc[ind]
                beta_max = StrategyData['Section'].ix[0].values
                indexes = []
                for i in StrategyData.index:
                    if np.any(beta_max < StrategyData['Section'].ix[i].values - .01):
                        # measure has sense at some point in time
                        beta_max = np.maximum(beta_max,StrategyData['Section'].ix[i].values - .01)
                        indexes.append(i)
                    else:
                        # inefficient measure
                        pass
                StrategyData = StrategyData.ix[indexes]
                StrategyData = StrategyData.sort_index()
            self.options[sec.name] = StrategyData.reset_index(drop=True)

    def evaluate(self,traject,solutions,OI_horizon=50,OI_year = 0):
        # cols = ['name'] + list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)
        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # measures at t=0 (2025) and t=20 (2045)
        if self.type == 'TC':
            #Step 2: calculate costs and risk reduction for each option
            #make a very basic traject dataframe of the current state that consists of current betas for each section per year
            BaseTrajectProbability = makeTrajectDF(traject,cols)
            count = 0
            measure_cols = ['Section','option_index','LCC','BC']
            TakenMeasures = pd.DataFrame(data= [[None, None, 0, None, None, None]], columns=measure_cols + ['name', 'params'])      #add columns (section name and index in self.options[section])

            #Calculate Total Cost for all sections (this does not change based on values later calculated so only has to be done once
            LCC = pd.DataFrame()
            for i in self.options:
                LCC_sec = pd.DataFrame(calcTC(self.options[i], r=self.r, horizon=cols[-1]), columns=[i])
                LCC = pd.concat([LCC, LCC_sec], ignore_index=False, axis=1)

            # Loop over measures
            Probability_steps = [copy.deepcopy(BaseTrajectProbability)]
            TrajectProbability = copy.deepcopy(BaseTrajectProbability)
            sections = list(self.options.keys())
            keys = copy.deepcopy(sections)
            while count < 80:
                count += 1
                print('Run number ' + str(count))
                PossibleMeasures = pd.DataFrame(columns=measure_cols)
                TotalCost = pd.DataFrame()
                BC = pd.DataFrame()
                dRs = [];TRs = []
                for i in self.options:
                    if i in keys:
                        # print('Considering section ' + i)
                        # somefunction(TC, TakenMeasures)
                        #assess the riskreduction based on the traject object (in a separate function to avoid alot of deepcopies) and the TakenMeasures
                        R_base, dR, TR = calcTR(i, self.options[i],TrajectProbability,original_section=TrajectProbability.loc[i], r=self.r,horizon=cols[-1],damage = traject.GeneralInfo['FloodDamage'])

                        # if count == 1:
                        #     cProfile.runctx("calcTR(i, self.options[i],TrajectProbability,original_section=TrajectProbability.loc[i], r=self.r,horizon=cols[-1],damage = traject.GeneralInfo['FloodDamage'])",globals=globals(),locals=locals())

                        #save TC
                        #if already a measure is in MeasuresTaken
                        #Reduce TotalCost by that LCC
                        if any(TakenMeasures['Section'].values==i):
                            LCC_done = TakenMeasures.loc[TakenMeasures['Section']==i]['LCC'].values
                        else:
                            LCC_done = 0
                        TotalCost_section = LCC[i][0:len(dR)]+TR-np.sum(LCC_done)
                        TotalCost = pd.concat([TotalCost, pd.DataFrame(TotalCost_section, columns=[i])], ignore_index=False, axis=1)

                        dRs.append(dR)
                        TRs.append(TR)

                        BC_section = np.divide(np.array(dR),np.array(LCC[i][0:len(dR)]).T)
                        #save BC
                        BC = pd.concat([BC, pd.DataFrame(BC_section, columns=[i])], ignore_index=False, axis=1)

                        ind = np.argmax(BC_section)
                        #pick option with highest BC ratio and add to PossibleMeasures
                        data_measure = pd.DataFrame([[i, ind, LCC[i][ind], BC_section[ind]]], columns=measure_cols)
                        PossibleMeasures = PossibleMeasures.append([data_measure],ignore_index=True)        #we can ditch this part
                        # add a routine that throws out all options that have a negative BC ratio in the first step
                        if count == 1:
                            pass
                    else:
                        print('Skipped section ' + i)
                #find section with highest BC
                indd = PossibleMeasures['Section'][PossibleMeasures['BC'].idxmax()]
                #find max BC of other sections
                maxBC_others = PossibleMeasures.nlargest(2,'BC').iloc[1].loc['BC']
                #make dataframe with cols TC and BC of section
                SectionMeasures = pd.concat([pd.DataFrame(TotalCost[indd].values, columns=['TotalCost']),
                                             pd.DataFrame(BC[indd].values, columns=['BC'])], axis=1)
                #select minimal TC for BC>BCothers
                if not any(SectionMeasures['BC']>1): break

                #The following is faster but less robust:
                id_opt = SectionMeasures.loc[SectionMeasures['BC'] > maxBC_others]['TotalCost'].idxmin()
                #This would be an in between:
                # if len(SectionMeasures.loc[SectionMeasures['BC'] > 3*maxBC_others])>2:
                #     id_opt = SectionMeasures.loc[SectionMeasures['BC'] > 3*maxBC_others]['TotalCost'].idxmin()
                # else:
                #     id_opt = SectionMeasures['BC'].idxmax()
                #This is most robust:
                # id_opt = SectionMeasures['BC'].idxmax()
                #Select the option with the highest BC ratio and add to TakenMeasures
                #The difference written is the difference in LCC!!!!
                LCCdiff = LCC[indd][id_opt] - np.sum(TakenMeasures.loc[TakenMeasures['Section']==indd]['LCC'].values)
                data_opt = pd.DataFrame([[indd, id_opt, LCCdiff, BC[indd][id_opt], self.options[indd].iloc[id_opt]['ID'].values[0], self.options[indd].iloc[id_opt]['params'].values[0]]],
                                        columns=measure_cols + ['ID', 'params'])                #here we evaluate and pick the option that has the lowest total cost and a BC ratio that is lower than any measure at any other section
                TakenMeasures = TakenMeasures.append(data_opt)

                #Update the TrajectProbability
                measuredata = self.options[indd].iloc[id_opt]
                TrajectProbability = ImplementOption(indd, TrajectProbability, measuredata)
                Probability_steps.append(copy.deepcopy(TrajectProbability))

                #Possible performance improvements

                #filter the sections to consider in the next step: only the sections with a reliability that is less than the weakest/some_factor e.g. 10 or 100.
                p_factor = 1000
                sectionlevel = TrajectProbability.xs('Section', level='mechanism')
                limit = norm.cdf(-np.min(sectionlevel.values.astype('float'), axis=0)) / p_factor
                indices = np.all(norm.cdf(-sectionlevel.values.astype('float')) > limit, axis=1)
                keys = []
                for i in range(0, len(indices)):
                    if indices[i]:
                        keys.append(sections[i])
            print('Run finished')
            self.Probabilities = Probability_steps
            self.TakenMeasures = TakenMeasures
        elif self.type == 'SmartOI':
            #find section where it is most attractive to make 1 or multiple mechanisms to meet the cross sectional reliability index
            #choice 1: geotechnical mechanisms ok for 2075

            #choice 2:also height ok for 2075
            pass
        elif self.type == 'OI':
            #compute cross sectional requirements
            N_piping = 1 + (traject.GeneralInfo['aPiping'] * traject.GeneralInfo['TrajectLength']/traject.GeneralInfo['bPiping'])
            N_stab = 1 + (traject.GeneralInfo['aStabilityInner'] * traject.GeneralInfo['TrajectLength'] / traject.GeneralInfo[
                'bStabilityInner'])
            N_overflow = 1
            beta_cs_piping = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaPiping']/ N_piping)
            beta_cs_stabinner = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaStabilityInner']/ N_stab)
            beta_cs_overflow = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaOverflow']/ N_overflow)

            #Rank sections based on 2075  Section probability
            beta_horizon = []
            for i in traject.Sections:
                beta_horizon.append(i.Reliability.SectionReliability.loc['Section'][str(OI_horizon)])
            section_indices = np.argsort(beta_horizon)

            measure_cols = ['Section','option_index','LCC','BC']
            TakenMeasures = pd.DataFrame(data= [[None, None, 0, None, None, None]], columns=measure_cols + ['name','params'])      #add columns (section name and index in self.options[section])
            BaseTrajectProbability = makeTrajectDF(traject,cols)
            Probability_steps = [copy.deepcopy(BaseTrajectProbability)]
            TrajectProbability = copy.deepcopy(BaseTrajectProbability)

            for j in section_indices:
                i = traject.Sections[j]
                #convert beta_cs to beta_section in order to correctly search self.options[section] THIS IS CURRENTLY INCONSISTENT WITH THE WAY IT IS CALCULATED
                beta_T_overflow = beta_cs_overflow
                beta_T_piping = -norm.ppf(norm.cdf(-beta_cs_piping)*(i.Length/traject.GeneralInfo['bPiping']))
                beta_T_stabinner = -norm.ppf(norm.cdf(-beta_cs_stabinner) * (i.Length / traject.GeneralInfo['bStabilityInner']))

                #find cheapest design that satisfies betatcs in 50 years from OI_year if OI_year is an int that is not 0
                if isinstance(OI_year,int):
                    targetyear = 50 #OI_year + 50
                else:
                    targetyear = 50
                #filter for overflow
                PossibleMeasures = copy.deepcopy(self.options[i.name].loc[self.options[i.name][('Overflow', targetyear)]  > beta_T_overflow])
                #filter for piping
                PossibleMeasures = PossibleMeasures.loc[self.options[i.name][('Piping', targetyear)]        > beta_T_piping]
                #filter for stabilityinner
                PossibleMeasures = PossibleMeasures.loc[PossibleMeasures[('StabilityInner', targetyear)]    > beta_T_stabinner]

                #calculate LCC
                LCC = calcTC(PossibleMeasures,r = self.r, horizon = self.options[i.name]['Overflow'].columns[-1])
                # select measure with lowest cost
                idx = np.argmin(LCC)
                measure = PossibleMeasures.iloc[idx]


                #calculate achieved risk reduction & BC ratio compared to base situation
                R_base, dR, TR = calcTR(i.name, measure, TrajectProbability,
                                        original_section=TrajectProbability.loc[i.name], r=self.r, horizon=cols[-1],damage=traject.GeneralInfo['FloodDamage'])
                BC = dR/LCC[idx]
                data_opt = pd.DataFrame([[i.name, idx, LCC[idx], BC, measure['ID'].values[0], measure['params'].values[0]]], columns=measure_cols + ['ID', 'params'])                #here we evaluate and pick the option that has the lowest total cost and a BC ratio that is lower than any measure at any other section

                #Add to TakenMeasures
                TakenMeasures = TakenMeasures.append(data_opt)
                #Calculate new probabilities
                TrajectProbability = ImplementOption(i.name, TrajectProbability, measure)
                Probability_steps.append(copy.deepcopy(TrajectProbability))
            self.TakenMeasures = TakenMeasures
            self.Probabilities = Probability_steps
            pass
        elif self.type == 'ExistingStrategy':
            # Input is a TestCaseStrategyTC object AND a to be considered TakenMeasures DataFrame (maybe some extra data is required)
            # We run through all the TakenMeasures, and each time compute the reliability

            pass

    def plotBetaTime(self,Traject, typ='single',fig_id = None,path = None, horizon = 100):
        step = 0
        beta_t = []
        plt.figure(100)
        for i in self.Probabilities:
            step += 1
            beta_t0, p = calcTrajectProb(i, horizon=horizon)
            beta_t.append(beta_t0)
            t = range(2025,2025+horizon+1)
            plt.plot(t, beta_t0, label=self.type+ ' stap ' + str(step))
        if typ == 'single':
            plt.plot([2025, 2025+horizon], [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])],
                     'k--', label='Norm')
            plt.xlabel('Time')
            plt.ylabel(r'$\beta$')
            plt.legend()
            plt.savefig(path + '\\' + 'figures' + '\\' 'BetaInTime' + self.type + '.png', bbox_inches='tight')
            plt.close(100)
        else:
            pass

    def plotBetaCosts(self, Traject, MeasureTable=None,
                      path = None, t = 0, cost_type = 'LCC',
                      typ='single', fig_id = None, last = 'no', labelmeasures='off', horizon = 100,
                      symbolmode = 'off',labels=None, linecolor='r',linestyle = '-',name = None, beta_or_prob='beta',
                      outputcsv=False):
        thislabel = labels
        if symbolmode == 'on':
            symbols = ['*','o','^','s','p','X','d','h','>','.','<','v','3','P','D']
            MeasureTable = MeasureTable.assign(symbol=symbols[0:len(MeasureTable)])

        if symbolmode == 'off': symbols = None
        if 'years' not in locals():
            years = Traject.Sections[0].Reliability.SectionReliability.columns.values.astype('float')
            horizon = np.max(years)
        step = 0
        betas = []
        pfs = []
        for i in self.Probabilities:
            step += 1
            beta_t0, p_t = calcTrajectProb(i, horizon=horizon)
            betas.append(beta_t0[t])
            pfs.append(p_t[t])
        # plot beta vs costs
        x = 0
        Costs = []
        if cost_type == 'LCC':
            for i in range(0, self.TakenMeasures['Section'].size):
                x += self.TakenMeasures['LCC'].iloc[i]
                Costs.append(x)
        elif cost_type == 'Initial':
            for i in range(0, self.TakenMeasures['Section'].size):
                if i > 0:
                    years = self.options[self.TakenMeasures.ix[i]['Section']].ix[self.TakenMeasures.ix[i]['option_index']]['year'].values[0]
                    if not isinstance(years, int):
                        for ij in range(len(years)):
                            if years[ij] == 0:
                                x += self.options[self.TakenMeasures.ix[i]['Section']].ix[
                                    self.TakenMeasures.ix[i]['option_index']]['cost'].values[0][ij]
                    elif isinstance(years, int):
                        if years > 0:
                            pass
                        else:
                            x += self.TakenMeasures['LCC'].iloc[i]
                    Costs.append(x)
                else:
                    Costs.append(x)

        Costs = np.divide(Costs,1e6)
        if typ == 'single':
            plt.figure(101)
            if symbolmode == 'off': plt.plot(Costs, betas, 'or-',label = self.type)
            if symbolmode == 'on':
                if beta_or_prob == 'beta': plt.plot(Costs, betas, 'r-',label = self.type)
                if beta_or_prob == 'prob': plt.plot(Costs, pfs, 'r-',label = self.type)
                print('add symbols for single plot')        # TO DO

            plt.plot([0, np.max(Costs)],
                     [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])], 'k--',
                     label='Norm')
            plt.xlabel('Total LCC in M€')
            if beta_or_prob == 'beta':
                plt.ylabel(r'$\beta$')
                plt.title('Total LCC versus ' + r'$\beta$' + ' in year ' + str(t + 2025))
                plt.savefig(path + '\\' + 'figures' + '\\' + 'BetavsCosts' + self.type + '_' + str(t + 2025) + '.png',
                            bbox_inches='tight')

            if beta_or_prob == 'prob':
                plt.ylabel(r'Failure probability $P_f$')
                plt.title('Total LCC versus ' + r'$P_f$' + ' in year ' + str(t + 2025))
                plt.savefig(path + '\\' + 'figures' + '\\' + 'PfvsCosts' + self.type + '_' + str(t + 2025) + '.png',
                            bbox_inches='tight')

            plt.close(101)
        else:
            plt.figure(fig_id)
            if beta_or_prob == 'beta':
                if labels != None:
                    if symbolmode == 'off': plt.plot(Costs, betas, 'o-', label= labels,color=linecolor, linestyle = linestyle)
                    if symbolmode == 'on': plt.plot(Costs, betas, '-', label= labels,color=linecolor, linestyle = linestyle,zorder=1)
                else:
                    if symbolmode == 'off': plt.plot(Costs, betas, 'o-',label = self.type,color=linecolor, linestyle = linestyle)
                    if symbolmode == 'on': plt.plot(Costs, betas, '-', label= self.type,color=linecolor, linestyle = linestyle,zorder=1)
            elif beta_or_prob == 'prob':
                if labels != None:
                    if symbolmode == 'off': plt.plot(Costs, pfs, 'o-', label= labels,color=linecolor, linestyle = linestyle)
                    if symbolmode == 'on': plt.plot(Costs, pfs, '-', label= labels,color=linecolor, linestyle = linestyle,zorder=1)
                else:
                    if symbolmode == 'off': plt.plot(Costs, pfs, 'o-',label = self.type,color=linecolor, linestyle = linestyle)
                    if symbolmode == 'on': plt.plot(Costs, pfs, '-', label= self.type,color=linecolor, linestyle = linestyle,zorder=1)
            if symbolmode == 'on':
                if beta_or_prob == 'beta':
                    interval = .07
                    base = np.max(betas) + interval
                    ycoord = np.array([base, base+interval, base+2*interval, base+3*interval])
                    ycoords = np.tile(ycoord,np.int(np.ceil(len(Costs)/len(ycoord))))
                elif beta_or_prob == 'prob':
                    interval = 2
                    base = np.min(pfs)/interval
                    ycoord = np.array([base, base/interval, base/(2 * interval), base/(3 * interval)])
                    ycoords = np.tile(ycoord, np.int(np.ceil(len(Costs) / len(ycoord))))
                for i in range(0,len(Costs)):
                    line = self.TakenMeasures.iloc[i]
                    if line['option_index'] != None:
                        if isinstance(line['ID'],list): line['ID'] = '+'.join(line['ID'])
                        if i> 0:
                            if Costs[i] > Costs[i-1]:
                                if beta_or_prob == 'beta':
                                    plt.scatter(Costs[i],betas[i],marker=MeasureTable.loc[MeasureTable['ID']==line['ID']]['symbol'].values[0],label=MeasureTable.loc[MeasureTable['ID']==line['ID']]['Name'].values[0],color=linecolor,edgecolors='k',linewidths=.5,zorder=2)
                                    plt.vlines(Costs[i],betas[i]+.05,ycoords[i]-.05,colors ='tab:gray', linestyles =':',zorder = 1)
                                elif beta_or_prob == 'prob':
                                    plt.scatter(Costs[i], pfs[i], marker=
                                    MeasureTable.loc[MeasureTable['ID'] == line['ID']]['symbol'].values[0],
                                                label=MeasureTable.loc[MeasureTable['ID'] == line['ID']]['Name'].values[
                                                    0], color=linecolor, edgecolors='k', linewidths=.5, zorder=2)
                                    plt.vlines(Costs[i], pfs[i], ycoords[i], colors='tab:gray',
                                               linestyles=':', zorder=1)
                                plt.text(Costs[i], ycoords[i], line['Section'][-2:],fontdict={'size':8},color = linecolor,horizontalalignment='center',zorder=3)

            if labelmeasures == 'on':
                for i in range(0, len(Costs)):
                    line = self.TakenMeasures.iloc[i]
                    if isinstance(line['option_index'],int):
                        meastyp = self.options[line['Section']].ix[line['option_index']]['type'].values[0]
                        if isinstance(meastyp,list):
                            meastyp = str(meastyp[0] + '+' + meastyp[1])
                    if i > 0:
                        if beta_or_prob == 'beta': plt.text(Costs[i], betas[i]-.1, line['Section'] + ': ' + meastyp)
                        if beta_or_prob == 'prob': plt.text(Costs[i], pfs[i]/2, line['Section'] + ': ' + meastyp)
            if last == 'yes':
                axes = plt.gca()
                xmax = np.max([axes.get_xlim()[1],np.max(Costs)])
                ceiling = np.ceil(np.max([xmax, np.max(Costs)]) / 10) * 10
                if beta_or_prob == 'beta': plt.plot([0, ceiling], [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])], 'k--', label='Norm')
                if beta_or_prob == 'prob': plt.plot([0, ceiling], [Traject.GeneralInfo['Pmax'],Traject.GeneralInfo['Pmax']], 'k--', label='Norm')
                if cost_type == 'LCC': plt.xlabel('Total LCC in M€')
                if cost_type == 'Initial': plt.xlabel('Cost in 2025 in M€')
                if beta_or_prob == 'beta': plt.ylabel(r'$\beta$')
                if beta_or_prob == 'prob': plt.ylabel(r'$P_f$')
                if beta_or_prob == 'prob':
                    axes.set_yscale('log')
                    axes.invert_yaxis()
                plt.xticks(np.arange(0, ceiling+1,10))
                axes.set_xlim(left = 0, right=ceiling)
                plt.grid()

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(),loc=4,fontsize ='x-small')

                # plt.legend(loc=5)

                if cost_type == 'LCC':
                    if beta_or_prob == 'beta': plt.title('Total LCC versus ' + r'$\beta$' + ' in year ' + str(t+2025))
                    if beta_or_prob == 'prob': plt.title('Total LCC versus ' + r'$P_f$' + ' in year ' + str(t + 2025))
                if cost_type == 'Initial':
                    if beta_or_prob == 'beta': plt.title('Costs versus ' + r'$\beta$' + ' in year ' + str(t + 2025))
                    if beta_or_prob == 'prob': plt.title('Costs versus ' + r'$P_f$' + ' in year ' + str(t + 2025))

                if beta_or_prob == 'beta': plt.title(r'Relation between $\beta$ and investment costs in M€')
                if beta_or_prob == 'prob': plt.title(r'Relation between $P_f$ and investment costs in M€')
                plt.xlabel('Investment costs in M€')

                if path is None:
                    plt.show()
                else:
                    if name is None:
                        if beta_or_prob == 'beta':
                            if cost_type == 'LCC': plt.savefig(path + '\\' + 'BetavsLCC_t' + str(t+2025) + '.png', bbox_inches='tight',dpi=300)
                            if cost_type == 'Initial': plt.savefig(path + '\\' + 'BetavsCosts_t' + str(t+2025) + '.png', bbox_inches='tight',dpi=300)
                        elif beta_or_prob == 'prob':
                            if cost_type == 'LCC': plt.savefig(path + '\\' + 'PfvsLCC_t' + str(t+2025) + '.png', bbox_inches='tight',dpi=300)
                            if cost_type == 'Initial': plt.savefig(path + '\\' + 'PfvsCosts_t' + str(t+2025) + '.png', bbox_inches='tight',dpi=300)
                    else:
                        if cost_type == 'LCC': plt.savefig(path + '\\LCC_' + name + str(t + 2025) + '.png', bbox_inches='tight',dpi=300)
                        if cost_type == 'Initial': plt.savefig(path + '\\Cost_' + name + str(t + 2025) + '.png', bbox_inches='tight',dpi=300)


                    plt.close(fig_id)

        if outputcsv:
            data = np.array([Costs.T, np.array(betas)]).T
            data = pd.DataFrame(data,columns=['Cost','beta'])
            if cost_type == 'LCC': data.to_csv(path + '\\' + 'BetavsLCC' + thislabel + '_t' + str(t+2025) + '.csv')
            if cost_type == 'Initial': data.to_csv(path + '\\' + 'BetavsCost' + thislabel + '_t' + str(t + 2025) + '.csv')
            pass

    def plotInvestmentSteps(self, TestCase,path=None,figure_size=(6,4)):
        updatedAssessment = copy.deepcopy(TestCase)
        for i in range(1,len(self.TakenMeasures)):
            if i == 6:
                print()
                pass
            updatedAssessment.plotReliabilityofDikeTraject(first=True, last=False,alpha=0.3,fig_size=figure_size)
            updatedAssessment.updateProbabilities(self.Probabilities[i],self.TakenMeasures.ix[i]["Section"])
            updatedAssessment.plotReliabilityofDikeTraject(pathname=path,first=False,
                                                           last=True,type=i,fig_size=figure_size)

        # for i in self.TakenMeasures:

    def writeProbabilitiesCSV(self, path,type):
        # with open(path + '\\ReliabilityLog_' + type + '.csv', 'w') as f:
        for i in range(0,len(self.Probabilities)):
            name = path + '\\ReliabilityLog_' + type + '_Step' + str(i) + '.csv'
            # measurerow = self.TakenMeasures.ix[i]['Section'] + ',' + self.TakenMeasures.ix[i]['name'] + ',' + str(self.TakenMeasures.ix[i]['params'])+ ',' + str(self.TakenMeasures.ix[i]['LCC'])
            # f.write(measurerow)

            # self.TakenMeasures.ix[i].to_csv(f, header=True)
            self.Probabilities[i].to_csv(path_or_buf=name, header=True)

def DetermineNewGeometry(geometry_change, slope_in, slope_out, direction, initial,bermheight = 2,geometry_plot = 'off',plot_dir = None):
    if len(initial) == 6:
        bermheight = initial.ix[2]['z']
    elif len(initial) == 4:
        pass
    #Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm  = geometry_change[1]

    geometry = initial.values
    cur_crest = np.max(geometry[:,1])
    new_crest = cur_crest+dcrest
    if geometry_plot == 'on':
        plt.plot(geometry[:, 0], geometry[:, 1],'k')
    if direction == 'outward':
        print("WARNING: outward reinforcement is NOT UP TO DATE!!!!")
        new_geometry = copy.deepcopy(geometry)
        for i in range(0,len(new_geometry)):
            #Run over points from the outside.
            if i >0:
                slope = (geometry[i - 1][1] - geometry[i][1]) / (geometry[i - 1][0] - geometry[i][0])
                if slope > 0 and new_geometry[i, 1] == cur_crest:  # outer slope
                    new_geometry[i][0] = (new_geometry[i][0] + dcrest / np.abs(slope))
                    new_geometry[i][1] = new_crest
                elif slope == 0:  # This is the crest
                    new_geometry[i][0] = new_geometry[i - 1][0] + np.abs(geometry[i - 1][0] - geometry[i][0])
                    new_geometry[i][1] = new_crest
                elif slope < 0 and new_geometry[i, 1] != cur_crest:  # This is the inner slope
                    new_geometry[i][0] = new_geometry[i - 1][0] + (
                                new_geometry[i - 1][1] - new_geometry[i][1]) / np.abs(slope)
                    # new_geometry[i][0]-(geometry[i-1][0] - new_geometry[i-1][0])
            else:
                new_geometry[i][0] = new_geometry[i][0]
        if dberm > 0:
            slope = (geometry[i - 1][1] - geometry[i][1]) / (geometry[i - 1][0] - geometry[i][0])
            x1 = new_geometry[i - 1][0] + (new_geometry[-2][1] - (new_geometry[-1][1] + bermheight)) / np.abs(slope)
            x2 = x1 + dberm
            y = bermheight + new_geometry[-1][1]
            new_geometry[-1][0] = new_geometry[-1][0] + dberm
            new_geometry = np.insert(new_geometry, [-1], np.array([x1, y]), axis=0)
            new_geometry = np.insert(new_geometry, [-1], np.array([x2, y]), axis=0)
            # add a shift
            for i in range(0, len(new_geometry)):
                new_geometry[i][0] -= dberm
        if geometry_plot == 'on':
            plt.plot(new_geometry[:, 0], new_geometry[:, 1], '--r')
    elif direction == 'inward':

        new_geometry = copy.deepcopy(geometry)
        # we start at the outer toe so reverse:
        for i in reversed(range(0,len(new_geometry)-1)):
            #preserve the outer slope
            slope = (geometry[i][1]-geometry[i + 1][1]) / (geometry[i + 1][0]-geometry[i][0])
            if slope > 0 and new_geometry[i, 1] == cur_crest:  # This is the outer slope
                new_geometry[i][0] = new_geometry[i][0] - dcrest / np.abs(slope) #point goes left with same slope
                new_geometry[i][1] = new_crest
            elif slope == 0:  # This is a horizontal part
                if geometry[i][1] == cur_crest:
                    new_geometry[i][0] = new_geometry[i + 1][0] - np.abs(geometry[i + 1][0] - geometry[i][0])
                    new_geometry[i][1] = new_crest
                elif geometry[i][1] == bermheight:
                    new_geometry[i][0] = new_geometry[i + 1][0] - np.abs(geometry[i + 1][0] - geometry[i][0]) - dberm
                    new_geometry[i][1] = bermheight
            elif slope < 0:  # This is the inner slope
                new_geometry[i][0] = new_geometry[i + 1][0] - (new_geometry[i + 1][1] - new_geometry[i][1]) * slope_in
        if dberm >0 and len(geometry) == 4:     #add a berm if the points dont exist yet
            x1 = new_geometry[0][0] + (bermheight*slope_in)
            x2 = x1 - dberm
            y = bermheight + new_geometry[0][1]

            new_geometry[0][0] = new_geometry[0][0] - dberm
            new_geometry = np.insert(new_geometry,[1], np.array([x1,y]),axis=0)
            new_geometry = np.insert(new_geometry,[1], np.array([x2,y]),axis=0)
        elif dberm > 0 and len(geometry) == 6:
            pass


    #calculate the area difference
    # geometry = np.append(geometry, np.array([new_geometry[-1]]), axis=0)
    area_old, polygon_old = calculateArea(geometry)
    area_new, polygon_new = calculateArea(new_geometry)
    poly_diff = polygon_new.difference(polygon_old)
    area_difference = poly_diff.area
    if geometry_plot == 'on':
        if hasattr(poly_diff,'geoms'):
            for i in range(0, len(poly_diff.geoms)):
                x1, y1 = poly_diff[i].exterior.xy
                plt.fill(x1, y1, 'r--')
        else:
            x1, y1 = poly_diff.exterior.xy
            plt.fill(x1, y1, 'r--')
        plt.text(np.mean(new_geometry[:, 0]), np.max(new_geometry[:, 1]),
                 'Area difference = ' + '{:.4}'.format(str(area_difference)) + ' $m^2$')
        if plot_dir == None:
            plt.show()
        else:
            plt.savefig(plot_dir + 'Geometry_'+ str(dberm) + '_' + str(dcrest) + '.png')
            plt.close()

    return new_geometry, area_difference


def calculateArea(geometry):
    #voeg virtueel punt onder de grond toe:
    extra = np.empty((1,2))
    if geometry[-1][1] > geometry[0][1]:
        extra[0,0] = geometry[-1][0]; extra[0,1] = geometry[0][1]
        geometry = np.append(geometry, np.array(extra),axis=0)
    elif geometry[-1][1] < geometry[0][1]:
        extra[0, 0] = geometry[0][0]; extra[0, 1] = geometry[-1][1]
        geometry = np.insert(geometry, [0], np.array(extra),axis=0)
    bottomlevel = np.min(geometry[:,1])
    area = 0
    for i in range(1,len(geometry)):
        a = np.abs(geometry[i-1][0]-geometry[i][0]) * (0.5 * np.abs(geometry[i-1][1]-geometry[i][1]) + 1.0 * (np.min((geometry[i-1][1],geometry[i][1])) - bottomlevel))
        area += a
    polypoints= []
    for i in range(0,len(geometry)):
        polypoints.append((geometry[i,0],geometry[i,1]))
    polygon = Polygon(polypoints)
    return area, polygon

def DetermineCosts(parameters, type, length, reinf_pars = None, housing = None, area_difference = None):
    if type == 'Soil reinforcement':
        if parameters['StabilityScreen'] == 'no':
            C = parameters['C_start'] + area_difference*parameters['C_unit'] * length
            if isinstance(housing, pd.DataFrame) and reinf_pars[1] >0.:
                C += parameters['C_house'] * housing.loc[float(reinf_pars[1])]['cumulative']
        elif parameters['StabilityScreen'] == 'yes':
            C = parameters['C_start'] + area_difference*parameters['C_unit'] * length + parameters['C_unit2'] * parameters['Depth'] * length
            if isinstance(housing, pd.DataFrame) and reinf_pars[1] >0.:
                C += parameters['C_house'] * housing.loc[float(reinf_pars[1])]['cumulative']
        #x = map(int, self.parameters['house_removal'].split(';'))
    elif type == 'Vertical Geotextile':
        C = parameters['C_unit'] * length
    elif type == 'Diaphragm Wall':
        C = parameters['C_unit'] * length
    elif type == 'Stability Screen':
        C = parameters['C_unit'] * parameters['Depth'] * length
    else:
        print('Unknown type')
    return C

def ProbabilisticDesign(design_variable, strength_input, Pt, horizon = 50, loadchange = 0, mechanism='Overflow'):
    if mechanism == 'Overflow':
        #determine the crest required for the target
        h_crest, beta = OverflowSimple(strength_input['h_crest'],strength_input['q_crest'], strength_input['h_c'],strength_input['q_c'],strength_input['beta'],mode='design',Pt=Pt,design_variable=design_variable)
        #add temporal changes due to settlement and climate change TO DO ADD CLIMATE
        h_crest = h_crest + horizon * (strength_input['dhc(t)'] + loadchange)
        return h_crest

def MeasureCombinations(combinables, partials,solutions):
    CombinedMeasures = pd.DataFrame(columns = combinables.columns)
    #loop over partials
    for i, row1 in partials.iterrows():
    #combine with all combinables
        for j, row2 in combinables.iterrows():
            ID = [row1['ID'].values[0],row2['ID'].values[0]]
            types = [row1['type'].values[0], row2['type'].values[0]]
            year = [row1['year'].values[0],row2['year'].values[0]]
            params = [row1['params'].values[0], row2['params'].values[0]]
            Cost = [row1['cost'].values[0],row2['cost'].values[0]]                #WARNING ADD DISCOUNTING IN EVALUATION OF MEASURES
            #combine betas
            #take maximums of mechanisms except if it is about StabilityInner for partial Stability Screen
            betas = []
            years = []
            for ij in partials.columns:
                if ij[0] != 'Section' and ij[1] != '':     #It is a beta value
                    # print('make something for stabilityscreen here')
                    # if ij[0] == 'StabilityInner' and row1['type'].values[0] == 'Stability Screen':
                    #     for ijk in solutions.Measures:
                    #         if ijk['ID'].values[0] == row2['ID'].values[0]:
                    #
                    #     #compute the LCR for the Soil Reinforcement with
                    #     print('adapt this')
                    #     #take the combinable
                    #     #add 0.2 to the parameters
                    # else:
                    beta = np.maximum(row1[ij],row2[ij])
                    years.append(ij[1])
                    betas.append(beta)
            #next update section probabilities
            for ij in partials.columns:
                if ij[0] =='Section':     #It is a beta value
                    #where year in years is the same as ij[1]
                    indices = [indices for indices, x in enumerate(years) if x == ij[1]]
                    ps = norm.cdf(-np.array(betas)[indices])
                    p = np.sum(ps)
                    betas.append(-norm.ppf(p))
            in1 = [ID, types, 'combined', year, params, Cost]
            allin = pd.DataFrame([in1 + betas], columns=combinables.columns)
            CombinedMeasures = CombinedMeasures.append(allin)
    return CombinedMeasures

def makeTrajectDF(traject,cols):
    # cols = cols[1:]
    sections = []
    for i in traject.Sections: sections.append(i.name)
    mechanisms = list(traject.Sections[0].MechanismData.keys()) + ['Section']
    df_index = pd.MultiIndex.from_product([sections, mechanisms], names=['name', 'mechanism'])
    TrajectProbability = pd.DataFrame(columns=cols, index=df_index)
    for i in traject.Sections:
        for j in mechanisms:
            TrajectProbability.loc[(i.name, j)] = list(i.Reliability.SectionReliability.loc[j])
    # TrajectProbability = TrajectProbability.append(pd.DataFrame(data,columns=cols))
    return TrajectProbability

def calcTC(section_options,r = 0.03, horizon = 100):
    TC = []
    #HIER VERDER MET INDEXING!
    for i,row in section_options.iterrows():
        if isinstance(row.loc['year'].values[0], list):
            year = row.loc['year'].values[0]
            cost = row.loc['cost'].values[0]
            totalcosts = 0
            for j in range(0, len(year)):
                totalcosts += cost[j] / ((1 + r) ** year[j])
            TC.append(float(totalcosts))

            # moeilijk lopen doen
        else:
            TC.append(float(row.loc['cost'] / ((1 + r) ** row.loc['year'])))

    return np.array(TC)

def calcTR(section, section_options, base_traject, original_section, r = 0.03, horizon = 100, damage = 1e9):
    #section: the section name
    #section_options: all options for the section
    #base_traject: traject probability with all implemented measures
    #takenmeasures: object with all measures taken
    #original section: series of probabilities of section, before taking a measure.
    if damage == 1e9: print('WARNING NO DAMAGE DEFINED')
    TotalRisk = []
    dR = []

    mechs = np.unique(base_traject.index.get_level_values('mechanism').values)
    sections = np.unique(base_traject.index.get_level_values('name').values)
    section_idx = np.where(sections==section)[0]
    section_options_array = {}
    base_array = {}
    TotalRisk = []; dR = []

    for i in mechs:
        base_array[i] = base_traject.xs(i,level=1).values.astype('float')
        if isinstance(section_options, pd.DataFrame):
            section_options_array[i] = section_options.xs(i,level=0,axis=1).values.astype('float')
            range_idx = len(section_options_array[mechs[0]])
        if isinstance(section_options, pd.Series):
            section_options_array[i] = section_options.xs(i,level=0).values.astype('float')
            range_idx = 0

    if 'section_options_array' in locals():
        base_risk = calcLifeCycleRisks(base_array, r, horizon, damage, datatype='Array',ts=base_traject.columns.values, mechs=mechs)
        for i in range(0,range_idx):
                TR = calcLifeCycleRisks(base_array, r, horizon, damage, change = section_options_array, section = section_idx, datatype = 'Array', ts=base_traject.columns.values, mechs=mechs, option=i)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)
    else:
        base_risk = calcLifeCycleRisks(base_traject, r, horizon, damage)
        if isinstance(section_options, pd.DataFrame):
            for i,row in section_options.iterrows():
                TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change = row, section = section)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)
        elif isinstance(section_options, pd.Series):
            TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change=section_options, section=section)
            TotalRisk.append(TR)
            dR.append(base_risk - TR)
    return base_risk, dR, TotalRisk

def calcLifeCycleRisks(base0,r, horizon,damage,change=None,section=None, datatype = 'DataFrame',ts = None,mechs = None, option = None):
    base = copy.deepcopy(base0)
    if datatype == 'DataFrame':
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        if isinstance(change,pd.Series):
            for i in mechs:
                #This is not very efficient. Could be improved.
                base.loc[(section, i)] = change.loc[i]
        else:
            pass
        beta_t, p_t = calcTrajectProb(base,horizon=horizon)
    elif datatype == 'Array':
        if isinstance(change,dict):
            for i in mechs:
                base[i][section] = change[i][option]
        else:
            pass
        beta_t, p_t = calcTrajectProb(base,horizon=horizon,datatype='Arrays',ts=ts,mechs=mechs)


    trange = np.arange(0, horizon + 1, 1)
    D_t = damage/(1+r)**trange
    risk_t = p_t*D_t
    TR = np.sum(risk_t)
    return TR

def calcTrajectProb(base,horizon=None, datatype = 'DataFrame',ts=None,mechs=None):
    pfs = {}
    if datatype == 'DataFrame':
        ts = base.columns.values
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        pf_traject = np.zeros((len(ts),))
        for i in mechs:
            if i != 'Section':
                betas = base.xs(i, level='mechanism').values.astype('float')
                pfs[i] = norm.cdf(-betas)
                pnonfs = 1-pfs[i]
                if i == 'Overflow':
                    pf_traject += np.max(pfs[i], axis=0)
                else:
                    # pf_traject += np.sum(pfs[i], axis=0)
                    pf_traject += 1-np.prod(pnonfs, axis = 0)

    elif datatype == 'Arrays':
        pf_traject = np.zeros((len(ts),))
        for i in mechs:
            if i!= 'Section':
                pfs[i] = norm.cdf(-base[i])
                pnonfs = 1-pfs[i]
                if i == 'Overflow':
                    pf_traject += np.max(pfs[i], axis=0)
                else:
                    # pf_traject += np.sum(pfs[i], axis=0)
                    pf_traject += 1-np.prod(pnonfs, axis = 0)
                    # print('old:' + str(np.sum(pfs[i], axis=0)))
                    # print('new:' + str(1-np.prod(pnonfs, axis = 0)))
    trange = np.arange(0,horizon+1,1)
    betafail = interp1d(ts,-norm.ppf(pf_traject))
    beta_t = betafail(trange)
    p_t = norm.cdf(-np.array(beta_t, dtype=np.float64))

    return beta_t,p_t



def ImplementOption(section,TrajectProbability,newProbability):
    mechs = np.unique(TrajectProbability.index.get_level_values('mechanism').values)
    #change trajectprobability by changing probability for each mechanism
    for i in mechs:
        TrajectProbability.loc[(section,i)] = newProbability[i]
    return TrajectProbability

