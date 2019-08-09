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
import cplex
from shapely.geometry import Polygon
import itertools
from DikeClasses import MechanismReliabilityCollection, SectionReliability
from HelperFunctions import IDtoName, flatten, pareto_frontier
import time
class Solutions:
    #This class contains possible solutions/measures
    def __init__(self, DikeSectionObject):
        self.SectionName = DikeSectionObject.name
        self.Length = DikeSectionObject.Length
        self.InitialGeometry = DikeSectionObject.InitialGeometry

    def fillSolutions(self,excelsheet):
        #read solutions from Excel
        data = pd.read_excel(excelsheet,'Measures')
        self.Measures = {}
        for i in data.index:
            self.Measures[i] = Measure(data.loc[i])
        self.MeasureTable = pd.DataFrame(columns=['ID', 'Name'])
        for i in range(0,len(self.Measures)):
            self.MeasureTable.loc[i] = [str(self.Measures[i].parameters['ID']), self.Measures[i].parameters['Name']]


    def evaluateSolutions(self,DikeSection,TrajectInfo,trange = [0,19,20,50,75,100], geometry_plot=False,plot_dir =
    None, preserve_slope = False):
        # Evaluate all the possible measures that are available:
        self.trange = trange
        removal = []
        for i in self.Measures:
            if self.Measures[i].parameters['available'] == 1:
                self.Measures[i].evaluateMeasure(DikeSection, TrajectInfo, geometry_plot=geometry_plot, plot_dir = plot_dir, preserve_slope = preserve_slope)
            else:
                removal.append(i)
        if len(removal) > 0:
            for i in removal:
                self.Measures.pop(i)

    def SolutionstoDataFrame(self, filtering='off',splitparams = False):
        #write all solutions to one single dataframe:
        mechanisms = list(self.Measures[list(self.Measures.keys())[0]].measures[0]['Reliability'].Mechanisms.keys()); mechanisms.append('Section')
        years = self.trange
        cols_r = pd.MultiIndex.from_product([mechanisms, years], names=['base', 'year'])
        reliability = pd.DataFrame(columns=cols_r)
        if splitparams:
            cols_m = pd.Index(['ID', 'type', 'class', 'year', 'yes/no', 'dcrest', 'dberm', 'cost'], name='base')
        else:
            cols_m = pd.Index(['ID', 'type', 'class', 'year', 'params', 'cost'], name='base')
        measure = pd.DataFrame(columns=cols_m)
        # data = pd.DataFrame(columns = cols)
        inputs_m = []
        inputs_r = []

        for i in list(self.Measures.keys()):
            if isinstance(self.Measures[i].measures, list):
                #if it is a list of measures (for soil reinforcement): write each entry of the list to the dataframe
                typee = self.Measures[i].parameters['Type']

                for j in range(len(self.Measures[i].measures)):
                    measure_in = []
                    reliability_in = []
                    if typee == 'Soil reinforcement':
                        designvars = ((self.Measures[i].measures[j]['dcrest'], self.Measures[i].measures[j]['dberm']))

                    cost = self.Measures[i].measures[j]['Cost']
                    measure_in.append(str(self.Measures[i].parameters['ID']))
                    measure_in.append(typee)
                    measure_in.append(self.Measures[i].parameters['Class'])
                    measure_in.append(self.Measures[i].parameters['year'])
                    if splitparams:
                        measure_in.append(-999)
                        measure_in.append(designvars[0])
                        measure_in.append(designvars[1])
                    else:
                        measure_in.append(designvars)
                    measure_in.append(cost)

                    betas = self.Measures[i].measures[j]['Reliability'].SectionReliability

                    for ij in mechanisms:
                        for ijk in betas.loc[ij].values:
                            reliability_in.append(ijk)

                    inputs_m.append(measure_in)
                    inputs_r.append(reliability_in)

            elif isinstance(self.Measures[i].measures, dict):
                ID = str(self.Measures[i].parameters['ID'])
                typee = self.Measures[i].parameters['Type']
                if typee == 'Vertical Geotextile':
                    designvars = self.Measures[i].measures['VZG']

                if typee == 'Diaphragm Wall':
                    designvars = self.Measures[i].measures['DiaphragmWall']

                classe = self.Measures[i].parameters['Class']
                yeare  = self.Measures[i].parameters['year']
                cost = self.Measures[i].measures['Cost']
                if splitparams:
                    inputs_m.append([ID, typee, classe, yeare, designvars, -999 , -999 ,cost])
                else:
                    inputs_m.append([ID, typee, classe, yeare, designvars, cost])
                betas = self.Measures[i].measures['Reliability'].SectionReliability
                beta = []
                for ij in mechanisms:
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)
        reliability = reliability.append(pd.DataFrame(inputs_r, columns=cols_r))
        measure = measure.append(pd.DataFrame(inputs_m, columns=cols_m))
        self.MeasureData = measure.join(reliability,how='inner')
        #fix multiindex:
        index = []
        for i in self.MeasureData.columns:
            index.append(i) if isinstance(i,tuple) else index.append((i,''))
        self.MeasureData.columns = pd.MultiIndex.from_tuples(index)
        if filtering == 'on': #here we could add some filtering on the measures, but it is not used right now.
            pass

    def plotBetaTimeEuro(self, measures='undefined',mechanism='Section',beta_ind = 'beta0',sectionname='Unknown',beta_req=None):
        # This function plots the relation between cost and beta in a certain year

        #measures is a list of measures that need to be plotted
        if measures == 'undefined':
            measures = list(self.Measures.keys())

        #mechanism can be used to select a single or all ('Section') mechanisms
        #beta can be used to use a criterion for selecting the 'best' designs, such as the beta at 't0'
        cols = ['type', 'parameters', 'Cost']
        [cols.append('beta' + str(i)) for i in self.trange]
        data = pd.DataFrame(columns=cols)
        num_plots = 5
        colors = sns.color_palette('hls', n_colors=num_plots)
        # colors = plt.cm.get_cmap(name=plt.cm.hsv, lut=num_plots)
        color = 0

        for i in np.unique(self.MeasureData['ID'].values):
            if isinstance(self.Measures[int(i) - 1].measures, list):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID'] == i])
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

                for j in range(len(cost_grid) - 1):
                    values = x.loc[(x['cost'] >= (cost_grid[j])) & (x['cost'] <= (cost_grid[j + 1]))][(mechanism, beta_ind)]
                    if len(list(values)) > 0:
                        idd = values.idxmax()
                        if betamax < np.max(list(values)):
                            betamax = np.max(list(values))
                            indices.append(idd)
                            if isinstance(x['cost'].loc[idd], pd.Series):
                                envelope_costs.append(x['cost'].loc[idd].values[0])

                            if not isinstance(x['cost'].loc[idd], pd.Series):
                                envelope_costs.append(x['cost'].loc[idd])

                            envelope_beta.append(betamax)

                if self.Measures[np.int(i)-1].parameters['Name'][-4:] != '2045':
                    plt.plot(envelope_costs, envelope_beta, color=colors[color], linestyle='-')
                    # [plt.text(y['Cost'].loc[ij], y[beta_ind].loc[i], y['parameters'].loc[ij],fontsize='x-small') for ij in indices]

                    plt.plot(y['cost'], y[(mechanism,beta_ind)], label = self.Measures[np.int(i)-1].parameters['Name'],
                             marker='o',markersize=6, color=colors[color],markerfacecolor=colors[color],
                             markeredgecolor=colors[color], linestyle='',alpha=1)

                    color += 1
            elif isinstance(self.Measures[np.int(i)-1].measures, dict):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData['ID'] == i])
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
    #class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    def __init__(self,inputs):
        self.parameters = {}
        for i in range(0,len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]
    def evaluateMeasure(self,DikeSection,TrajectInfo, geometry_plot=False, plot_dir = None,preserve_slope = False):

        from HelperFunctions import createDir

        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        SFincrease = 0.2        #for stability screen
        if geometry_plot:
            plt.figure(1000)

        #different types of measures:
        if type == 'Soil reinforcement':
            createDir(plot_dir)
            crest_step = 0.5
            berm_step = 10
            crestrange = np.linspace(self.parameters['dcrest_min'], self.parameters['dcrest_max'], 1 + (self.parameters['dcrest_max']-self.parameters['dcrest_min']) / crest_step)
            if self.parameters['Direction'] == 'outward':
                bermrange = np.linspace(0., self.parameters['max_outward'], 1+(self.parameters['max_outward']/berm_step))
            elif self.parameters['Direction'] == 'inward':
                bermrange = np.linspace(0., self.parameters['max_inward'], 1+(self.parameters['max_inward']/berm_step))
            measures = [[x,y] for x in crestrange for y in bermrange]
            if not preserve_slope:
                slope_in = 4
                slope_out = 3 #inner and outer slope
            else:
                slope_in = False
                slope_out = False

            self.measures = []
            if self.parameters['StabilityScreen'] == 'yes':
                self.parameters['Depth'] = DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input['d_cover'] + 1.

            for j in measures:
                self.measures.append({})
                self.measures[-1]['dcrest'] =j[0]
                self.measures[-1]['dberm'] = j[1]
                self.measures[-1]['Geometry'], area_difference = DetermineNewGeometry(j,self.parameters['Direction'],DikeSection.InitialGeometry,geometry_plot=geometry_plot, plot_dir = plot_dir, slope_in = slope_in)
                self.measures[-1]['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, reinf_pars = j, housing = DikeSection.houses, area_difference= area_difference)
                self.measures[-1]['Reliability'] = SectionReliability()
                self.measures[-1]['Reliability'].Mechanisms = {}

                for i in mechanisms:
                    calc_type = DikeSection.MechanismData[i][1]
                    self.measures[-1]['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=TrajectInfo['T'], measure_year=self.parameters['year'])
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

                                elif self.parameters['Direction'] == 'outward': #not implemented
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] = self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] \
                                                                                                            + (self.measures[-1]['dberm'] * self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['dSF/dberm'])
                            elif i == 'Piping':
                                self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lvoor'] = \
                                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lvoor'] + self.measures[-1]['dberm']
                                self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lachter'] = \
                                    np.max([0.,self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lachter'] - self.measures[-1]['dberm']])
                    self.measures[-1]['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
                self.measures[-1]['Reliability'].calcSectionReliability(TrajectInfo,DikeSection.Length)
        elif type == 'Vertical Geotextile':
            #No influence on overflow and stability
            #Only 1 parameterized version with a lifetime of 50 years

            self.measures = {}
            self.measures['VZG'] = 'yes'
            self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
            self.measures['Reliability'] = SectionReliability()
            self.measures['Reliability'].Mechanisms = {}

            for i in mechanisms:
                calc_type = DikeSection.MechanismData[i][1]
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=TrajectInfo['T'])
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
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=TrajectInfo['T'])
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
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, years=TrajectInfo['T'])
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
    #define a strategy object to evaluate measures in accordance with a strategy: options are:
    # TC/Heuristic
    # OI/TargetReliability
    # MultiInteger (under dev.)
    # GeneticAlgorithm (under dev.)
    def __init__(self, type, r=0.03):
        self.type = type        #OI or CB
        self.r = r
        if type == 'MultiInteger' or type == 'GeneticAlgorithm':
            pass

    def combine(self, traject, solutions, filtering='off',splitparams=False, OI_horizon=50, OI_year=0):
        #This routine combines 'combinable' solutions to options with two measures (e.g. VZG + 10 meter berm)
        self.options = {}

        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # measures at t=0 (2025) and t=20 (2045)
        for i in range(0, len(traject.Sections)):
            sec = traject.Sections[i]
            # Step 1: combine measures with partial measures
            combinables = solutions[traject.Sections[i].name].MeasureData.loc[solutions[traject.Sections[i].name].MeasureData['class'] == 'combinable']
            if self.type == 'OI' and isinstance(OI_year, int):
                combinables = combinables.loc[solutions[traject.Sections[i].name].MeasureData['year'] == OI_year]

            partials = solutions[traject.Sections[i].name].MeasureData.loc[solutions[traject.Sections[i].name].MeasureData['class'] == 'partial']
            if self.type == 'OI' and isinstance(OI_year, int):
                partials = partials.loc[solutions[traject.Sections[i].name].MeasureData['year'] == OI_year]

            combinedmeasures = MeasureCombinations(combinables, partials, solutions[traject.Sections[i].name],
                                                   splitparams=splitparams)
            # make sure combinable, mechanism and year are in the MeasureData dataframe
            # make a strategies dataframe where all combinable measures are combined with partial measures for each timestep
            #if there is a measureid that is not known yet, add it to the measure table

            existingIDs = solutions[traject.Sections[i].name].MeasureTable['ID'].values
            IDs = np.unique(combinedmeasures['ID'].values)
            if len(IDs) > 0:
                for ij in IDs:
                    if ij not in existingIDs:
                        indexes = ij.split('+')
                        name = solutions[traject.Sections[i].name].MeasureTable.loc[solutions[traject.Sections[
                            i].name].MeasureTable['ID'] == indexes[0]]['Name'].values[0] + \
                               '+' + solutions[traject.Sections[i].name].MeasureTable.loc[solutions[traject.Sections[
                            i].name].MeasureTable['ID'] == indexes[1]]['Name'].values[0]
                        solutions[traject.Sections[i].name].MeasureTable.loc[len(solutions[traject.Sections[
                            i].name].MeasureTable) + 1] = name
                        solutions[traject.Sections[i].name].MeasureTable.loc[len(solutions[traject.Sections[
                            i].name].MeasureTable)]['ID'] = ij

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

                for j in StrategyData.index:
                    if np.any(beta_max < StrategyData['Section'].ix[j].values - .01):
                        # measure has sense at some point in time
                        beta_max = np.maximum(beta_max, StrategyData['Section'].ix[j].values - .01)
                        indexes.append(i)
                    else:
                        # inefficient measure
                        pass

                StrategyData = StrategyData.ix[indexes]
                StrategyData = StrategyData.sort_index()

            self.options[sec.name] = StrategyData.reset_index(drop=True)

    def evaluate(self, traject, solutions, OI_horizon=50, OI_year=0, splitparams = False):
        #This is the core code of the optimization. This piece should probably be split into the different methods available.

        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # measures at t=0 (2025) and t=20 (2045)
        if self.type == 'TC' or self.type == 'Heuristic':
            #Step 2: calculate costs and risk reduction for each option
            #make a very basic traject dataframe of the current state that consists of current betas for each section per year
            BaseTrajectProbability = makeTrajectDF(traject, cols)
            count = 0
            measure_cols = ['Section', 'option_index', 'LCC', 'BC']
            if splitparams:
                TakenMeasures = pd.DataFrame(data=[[None, None, 0, None,None , None, None,None,None]],
                                             columns=measure_cols + ['ID','name','yes/no','dcrest','dberm'])
                # add columns (section name and index in self.options[section])
            else:
                TakenMeasures = pd.DataFrame(data=[[None,None, None, 0, None, None, None]],
                                             columns=measure_cols + ['ID','name', 'params'])

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

            while count < 80: #counter set to 80 to limit calculations. But usually it stops much earlier as the BC-ratio drops below 1
                count += 1
                print('Run number ' + str(count))
                PossibleMeasures = pd.DataFrame(columns=measure_cols)
                TotalCost = pd.DataFrame()
                BC = pd.DataFrame()
                dRs = [];TRs = []

                for i in self.options:
                    if i in keys:
                        #assess the riskreduction based on the traject object (in a separate function to avoid alot of deepcopies) and the TakenMeasures
                        R_base, dR, TR = calcTR(i, self.options[i], TrajectProbability, original_section=TrajectProbability.loc[i], r=self.r, horizon=cols[-1], damage=traject.GeneralInfo['FloodDamage'])

                        #save TC
                        #if already a measure is in MeasuresTaken
                        #Reduce TotalCost by that LCC
                        if any(TakenMeasures['Section'].values == i):
                            LCC_done = TakenMeasures.loc[TakenMeasures['Section'] == i]['LCC'].values
                        else:
                            LCC_done = 0

                        TotalCost_section = LCC[i][0: len(dR)] + TR - np.sum(LCC_done)
                        TotalCost = pd.concat([TotalCost, pd.DataFrame(TotalCost_section, columns=[i])], ignore_index=False, axis=1)
                        dRs.append(dR)
                        TRs.append(TR)
                        BC_section = np.divide(np.array(dR), np.array(LCC[i][0:len(dR)]).T)

                        #save BC
                        BC = pd.concat([BC, pd.DataFrame(BC_section, columns=[i])], ignore_index=False, axis=1)
                        ind = np.argmax(BC_section)

                        #pick option with highest BC ratio and add to PossibleMeasures
                        data_measure = pd.DataFrame([[i, ind, LCC[i][ind], BC_section[ind]]], columns=measure_cols)
                        PossibleMeasures = PossibleMeasures.append([data_measure], ignore_index=True)

                        # TODO: add a routine that throws out all options that have a negative BC ratio in the first step
                        if count == 1:
                            pass

                    else:
                        print('Skipped section ' + i)

                #find section with highest BC
                indd = PossibleMeasures['Section'][PossibleMeasures['BC'].idxmax()]

                #find max BC of other sections
                maxBC_others = PossibleMeasures.nlargest(2, 'BC').iloc[1].loc['BC']

                #make dataframe with cols TC and BC of section
                SectionMeasures = pd.concat([pd.DataFrame(TotalCost[indd].values, columns=['TotalCost']), pd.DataFrame(BC[indd].values, columns=['BC'])], axis=1)

                #select minimal TC for BC>BCothers
                if not any(SectionMeasures['BC'] > 1):
                    break

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
                LCCdiff = LCC[indd][id_opt] - np.sum(TakenMeasures.loc[TakenMeasures['Section'] == indd]['LCC'].values)
                if splitparams:
                    data_opt = pd.DataFrame([[indd, id_opt, LCCdiff,
                                              BC[indd][id_opt], self.options[indd].iloc[id_opt]['ID'].values[0],
                                              IDtoName(self.options[indd].iloc[id_opt][
                                                                             'ID'].values[0],solutions[
                                                  indd].MeasureTable),
                                              self.options[indd].iloc[id_opt]['yes/no'].values[0],self.options[indd].iloc[id_opt]['dcrest'].values[0],self.options[indd].iloc[id_opt]['dberm'].values[0]]],
                                            columns=TakenMeasures.columns)
                else:
                    data_opt = pd.DataFrame([[indd, id_opt, LCCdiff, BC[indd][id_opt], self.options[indd].iloc[
                        id_opt]['ID'].values[0], IDtoName(self.options[indd].iloc[id_opt]['ID'].values[0],solutions[
                        indd].MeasureTable), self.options[indd].iloc[id_opt]['params'].values[0]]],
                                            columns=TakenMeasures.columns)

                #here we evaluate and pick the option that has the lowest total cost and a BC ratio that is lower than any measure at any other section
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
        elif self.type == 'SmartOI' or self.type == 'SmartTargetReliability':
            # TODO add a smarter OI version where the failure probability budget is partially redistributed of the mechanisms.

            #find section where it is most attractive to make 1 or multiple mechanisms to meet the cross sectional reliability index
            #choice 1: geotechnical mechanisms ok for 2075

            #choice 2:also height ok for 2075
            pass
        elif self.type == 'OI' or self.type == 'TargetReliability':
            #compute cross sectional requirements
            N_piping = 1 + (traject.GeneralInfo['aPiping'] * traject.GeneralInfo['TrajectLength'] / traject.GeneralInfo['bPiping'])
            N_stab = 1 + (traject.GeneralInfo['aStabilityInner'] * traject.GeneralInfo['TrajectLength'] / traject.GeneralInfo['bStabilityInner'])
            N_overflow = 1
            beta_cs_piping = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaPiping'] / N_piping)
            beta_cs_stabinner = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaStabilityInner'] / N_stab)
            beta_cs_overflow = -norm.ppf(traject.GeneralInfo['Pmax']*traject.GeneralInfo['omegaOverflow'] / N_overflow)

            #Rank sections based on 2075 Section probability
            beta_horizon = []

            for i in traject.Sections:
                beta_horizon.append(i.Reliability.SectionReliability.loc['Section'][str(OI_horizon)])

            section_indices = np.argsort(beta_horizon)

            measure_cols = ['Section', 'option_index', 'LCC', 'BC']
            TakenMeasures = pd.DataFrame(data=[[None, None, 0, None, None, None]], columns=measure_cols + ['name', 'params'])      #add columns (section name and index in self.options[section])
            BaseTrajectProbability = makeTrajectDF(traject, cols)
            Probability_steps = [copy.deepcopy(BaseTrajectProbability)]
            TrajectProbability = copy.deepcopy(BaseTrajectProbability)

            for j in section_indices:
                i = traject.Sections[j]
                #convert beta_cs to beta_section in order to correctly search self.options[section] THIS IS CURRENTLY INCONSISTENT WITH THE WAY IT IS CALCULATED
                beta_T_overflow = beta_cs_overflow
                beta_T_piping = -norm.ppf(norm.cdf(-beta_cs_piping) * (i.Length / traject.GeneralInfo['bPiping']))
                beta_T_stabinner = -norm.ppf(norm.cdf(-beta_cs_stabinner) * (i.Length / traject.GeneralInfo['bStabilityInner']))

                #find cheapest design that satisfies betatcs in 50 years from OI_year if OI_year is an int that is not 0
                if isinstance(OI_year, int):
                    targetyear = 50 #OI_year + 50
                else:
                    targetyear = 50

                #filter for overflow
                PossibleMeasures = copy.deepcopy(self.options[i.name].loc[self.options[i.name][('Overflow', targetyear)] > beta_T_overflow])

                #filter for piping
                PossibleMeasures = PossibleMeasures.loc[self.options[i.name][('Piping', targetyear)] > beta_T_piping]

                #filter for stabilityinner
                PossibleMeasures = PossibleMeasures.loc[PossibleMeasures[('StabilityInner', targetyear)] > beta_T_stabinner]

                #calculate LCC
                LCC = calcTC(PossibleMeasures, r=self.r, horizon=self.options[i.name]['Overflow'].columns[-1])

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
        elif self.type == 'ParetoFrontier':
            #make indices of all combinations:
            option_sizes = []
            start =time.time()
            for i in self.options.keys():
                option_sizes.append(range(0,np.size(self.options[i],0)))
            option_combis = list(itertools.product(*option_sizes))
            print('Number of combinations: ' + str(len(option_combis)))
            print('Finding all option combis: ' + str(time.time()-start))
            if len(option_combis)>10000:
                ind = np.random.choice(range(0,len(option_combis)),size = 10000,replace=False)
                option_combis = list(np.array(option_combis)[ind])
            #compute the investment cost of all combinations

            start = time.time()
            LCC = np.empty((np.max(option_combis)+1, len(self.options)))
            LCC.fill(1e99)
            count = 0
            for i in self.options.keys():
                LCC[:len(self.options[i]), count] = calcTC(self.options[i], self.r,
                                       np.max(traject.GeneralInfo['T']))
                count += 1
            print('Calculating individual LCCs (n=10000): ' + str(time.time()-start))
            start = time.time()
            option_combi_LCC = []
            for i in option_combis:
                LCCsum = 0
                for j in range(0,len(i)):
                    LCCsum += LCC[i[j],j]
                option_combi_LCC.append(LCCsum)
            print('Compute all LCC: (n=10000): ' + str(time.time()-start))
            # option_combi_LCC = [LCC[i[0],0]+LCC[i[1],1] for i in option_combis] FASTER BUT HARD TO MAKE SIZE
            # INDEPNDENT

            #compute the risk for all combinations of measures
            #loop over all option combis
            #we make a probability object based on the option_combi indices
            start = time.time()
            TotalRisk = []
            time_LCRcalc = []
            time_writeprobs = []
            probability_array = {}
            for mech in traject.GeneralInfo['Mechanisms']:
                # make an n,s,t array for each mecahnism
                probability_array[mech] = np.empty(
                    (len(self.options.keys()), np.max(option_combis) + 1, len(traject.GeneralInfo['T'])))
                for section in range(0, len(self.options.keys())):
                    secname = list(self.options.keys())[section]
                    probability_array[mech][section, :len(self.options[secname]), :] = self.options[secname][mech].values

            for i in option_combis:
                probs = {}
                for mech in traject.GeneralInfo['Mechanisms']:
                    probs[mech] = np.empty((len(self.options), len(traject.GeneralInfo['T'])))
                    count = 0
                    for sec in range(0,len(self.options.keys())):
                        start3 = time.time()
                        probs[mech][count, :] = probability_array[mech][sec,i[count],:]
                        time_writeprobs.append(time.time()-start3)
                        count += 1
                start2 = time.time()
                TotalRisk.append(calcLifeCycleRisks(probs, self.r, np.max(traject.GeneralInfo['T']),
                                                    traject.GeneralInfo['FloodDamage'], datatype='Array',
                                                    ts=traject.GeneralInfo['T']))
                time_LCRcalc.append(time.time()-start2)
            print('Compute risk: (n=10000): ' + str(time.time()-start))
            print('Total time for LCR calculation: ' + str(np.sum(time_LCRcalc)))
            print('Total time for reading probabilities: ' + str(np.sum(time_writeprobs)))
            self.option_combis = option_combis
            self.LCC_combis = option_combi_LCC
            self.TotalRisk_combis = TotalRisk
                #then we run a sc
            # ImplementOption()
            # calcTR()
            # pass

    def filter(self,traject, type='ParetoPerSection'):
        # self.options_height, self.options_geotechnical = split_options(self.options)
        if type == 'ParetoPerSection':
            damage = traject.GeneralInfo['FloodDamage']
            r = self.r
            horizon = np.max(traject.GeneralInfo['T'])
            self.options_filtered = copy.deepcopy(self.options)
            #we filter the options for each section, such that only interesting ones remain
            for i in self.options.keys():

                #indexes part 1: only the pareto front for stability and piping
                LCC =calcTC(self.options[i])

                tgrid = self.options[i]['StabilityInner'].columns.values
                pf1 = norm.cdf(-self.options[i]['Section'])
                pftot1 = interp1d(tgrid, pf1)
                risk1 = np.sum(pftot1(np.arange(0, horizon, 1)) * (damage / (1 + r) ** np.arange(0, horizon, 1)),axis=1)
                paretolcc,paretorisk,index1 = pareto_frontier(LCC,risk1,maxX=False,maxY=False)
                index= index1

                # pf1 = norm.cdf(-self.options[i]['StabilityInner']) + norm.cdf(-self.options[i]['Piping'])
                # pftot2 = interp1d(tgrid, pf2)
                # risk2 = np.sum(pftot2(np.arange(0, horizon, 1)) * (damage / (1 + r) ** np.arange(0, horizon, 1)),axis=1)
                # paretolcc,paretorisk,index2 = pareto_frontier(LCC,risk2,maxX=False,maxY=False)
                # index = index1 + list(set(index2)-set(index1))

                self.options_filtered[i] = self.options_filtered[i].iloc[index]
                self.options_filtered[i]['LCC'] = LCC[index]
                self.options_filtered[i] = self.options_filtered[i].reset_index(drop=True)
                print('For dike section ' + i + ' reduced size from ' + str(len(LCC)) + ' to ' + str(len(index)))

                # plt.plot(LCC, risk, 'xr')
                # plt.plot(paretolcc, paretorisk, 'xb')
                # plt.plot(LCC[index],risk[index], 'dg')
                # plt.show()

                #indexes part 2: only the pareto front for section probability






            #swap filtered and original measures:
            self.options_old = copy.deepcopy(self.options)
            self.options = copy.deepcopy(self.options_filtered)
            del self.options_filtered

    def make_optimization_input(self, traject,solutions):
        self.options_height, self.options_geotechnical = split_options(self.options)


        N = len(self.options)                               # Number of dike sections
        T = np.max(traject.GeneralInfo['T'])                # Number of time steps
        Sh = 0
        Sg = 0
        # #Number of strategies (maximum for all dike sections), for geotechnical and height
        Sh = np.max([np.max([Sh, np.max(len(self.options_height[i]))]) for i in self.options_height.keys()])
        Sg = np.max([np.max([Sg, np.max(len(self.options_geotechnical[i]))]) for i in self.options_geotechnical.keys()])

        #probabilities [N,S,T]
        mechs = traject.GeneralInfo['Mechanisms']
        self.Pf = {}
        for i in mechs:
            if i == 'Overflow':
                self.Pf[i]     = np.full((N, Sh+1, T),1.)
            else:
                self.Pf[i]     = np.full((N, Sg + 1, T), 1.)
            #old:
            # self.Pf[i]     = np.full((N, S, T), 1.)
        #fill values
        # TODO Think about the initial condition and whether this should be added separately or teh 0,
        #  0 soil reinforcement also suffices.
        keys = list(self.options.keys())

        #get all probabilities. Interpolate on beta per section, then combine p_f
        betas = {}
        for n in range(0,N):
            for i in mechs:
                len_beta1 = traject.Sections[n].Reliability.SectionReliability.shape[1]
                beta1 = traject.Sections[n].Reliability.SectionReliability.loc[i].values.reshape((len_beta1, 1)).T #Initial
                # condition with no measure
                if i == 'Overflow':
                    beta2 = self.options_height[keys[n]][i]
                    #All solutions
                else:
                    beta2 = self.options_geotechnical[keys[n]][i] # All solutions
                betas[i] = np.concatenate((beta1, beta2),axis=0)
                if np.shape(betas[i])[1] != T:
                    betas[i] = interp1d(traject.GeneralInfo['T'], betas[i])(np.arange(0, T, 1))
                self.Pf[i][n,0:np.size(betas[i],0),:]      = norm.cdf(-betas[i])


        # Costs of options [N,Sh,Sg]
        self.LCCOption = np.full((N, Sh + 1, Sg + 1), 1e99)
        for n in range(0, len(keys)):
            self.LCCOption[n, 0, 0] = 0.
            LCC_sh = calcTC(self.options_height[keys[n]])
            LCC_sg = calcTC(self.options_geotechnical[keys[n]])
            # LCC_tot = calcTC(self.options[keys[n]])
            for sh in range(0, len(self.options_height[keys[n]])):
                #if it is a full type, it should only be combined with another full of the same type
                if (self.options_height[keys[n]].iloc[sh]['class'].values[0] == 'full'):
                    full = True
                else:
                    full = False
                # if (self.options_height[keys[n]].iloc[sh]['type'].values[0] == 'Diaphragm Wall') | (
                #         self.options_height[keys[n]].iloc[sh]['type'].values[0] == 'Stability Screen'):
                #     full_structure = True
                # else:
                #     full_structure = False
                for sg in range(0, len(self.options_geotechnical[keys[n]])):# Sg):
                    #if sh is a diaphragm wall, only a diaphragm wall can be taken for sg
                    if full:
                        #if the type is different it is not a possibility:
                        if self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] != self.options_height[keys[
                            n]].iloc[sh]['type'].values[0]:
                            pass  # do not change value, impossible combination (keep at 1e99)
                        #if it is a stability screen, use same reasoning as for single Vertical Geotextile
                        elif self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] == 'Stability Screen':
                            self.LCCOption[n, 0, sg + 1] = LCC_sg[sg]
                        else:
                            #if the type is a soil reinforcement, the year has to be the same
                            if (self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] == 'Soil reinforcement'):
                                if (self.options_geotechnical[keys[n]].iloc[sg]['year'].values[0] ==self.options_height[keys[n]].iloc[sh]['year'].values[0]) & \
                                        (self.options_geotechnical[keys[n]].iloc[sg]['class'].values[0] == 'full'):
                                    self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[sh] + LCC_sg[sg]  # only use the costs once
                                else:
                                    pass #not allowed
                            else:    #Diaphragm wall
                                self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[sh]  # only use the costs once
                    #if sg is a plain geotextile, it can only be combined with no measure for height, otherwise it
                    # would be combined (should be extended to stability screen)
                    elif self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] == 'Vertical Geotextile':
                        #can only be combined with no measure for height
                        self.LCCOption[n, 0, sg+1] = LCC_sg[sg]
                    #if sg is a combined measure the soil reinforcement timing should be aligned:
                    elif self.options_geotechnical[keys[n]].iloc[sg]['class'].values[0] == 'combined':
                        #check if the time of the soil reinforcement part is the same as for the height one
                        #first extract the index of the soil reinforcement
                        index = np.argwhere(np.array(self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0]) ==
                                            'Soil reinforcement')[0][0]
                        if self.options_geotechnical[keys[n]].iloc[sg]['year'].values[0][index] \
                                ==self.options_height[keys[n]].iloc[sh]['year'].values[0]:
                            self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[sh] + LCC_sg[sg]  # only use the costs once
                        else:
                            pass #dont change, impossible combi
                        #if sg is combinable, it should only be combined with sh that have the same year
                    elif self.options_geotechnical[keys[n]].iloc[sg]['class'].values[0] == 'combinable':
                        if self.options_geotechnical[keys[n]].iloc[sg]['year'].values[0] == self.options_height[keys[
                            n]].iloc[sh]['year'].values[0]:
                            self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[sh] + LCC_sg[sg]  # only use the costs once
                        else:
                            pass
                    elif self.options_geotechnical[keys[n]].iloc[sg]['class'].values[0] == 'full':
                        pass # not allowed as the sh is not 'full'
                    else:
                        # if sg is a diaphragm wall cost should be only accounted for once:
                        if self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] != 'Diaphragm Wall':
                            self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[sh] + LCC_sg[sg]  # only use the costs once
                        else:
                            pass

        #Decision Variables for executed options [N,Sh] & [N,Sg]
        self.Cint_h           = np.zeros((N,Sh))
        self.Cint_g           = np.zeros((N,Sg))

        #Decision Variable for weakest overflow section with dims [N,Sh]
        self.Dint           = np.zeros((N,Sh))

        #add discounted damage [T,]
        self.D = np.array(traject.GeneralInfo['FloodDamage'] * (1 / ((1 + self.r) ** np.arange(0, T, 1))))

        #expected damage for overflow and for piping & slope stability
        # self.RiskGeotechnical = np.zeros((N,Sg+1,T))
        self.RiskGeotechnical = (self.Pf['StabilityInner'] + self.Pf['Piping']) * np.tile(self.D.T,(N,Sg+1,1))

        self.RiskOverflow = self.Pf['Overflow'] *np.tile(self.D.T,(N,Sh+1,1)) #np.zeros((N,Sh+1,T))
        #add a few general parameters
        self.opt_parameters = {'N':N, 'T':T, 'Sg':Sg+1 ,'Sh':Sh+1}
    def create_optimization_model(self):
        #make a model
        #enlist all the variables
        model = cplex.Cplex()
        grN  = range(self.opt_parameters['N'])
        # grS = range(self.opt_parameters['S'])
        grSh = range(self.opt_parameters['Sh'])
        grSg = range(self.opt_parameters['Sg'])
        grT  = range(self.opt_parameters['T'])

        #all variables
        Cint_nd = np.array([[["C" + str(n).zfill(3) + str(sh).zfill(3) + str(sg).zfill(3) for sg in grSg] for sh in grSh] for n in grN])
        Dint_nd = np.array([[["D" + str(n).zfill(3) + str(s).zfill(3) + str(t).zfill(3) for t in grT] for s in grSh] for n in grN])

        #names of variables
        Cint = ["C" + str(n).zfill(3) + str(sh).zfill(3) +str(sg).zfill(3) for sg in grSg for sh in grSh for n in grN]
        Dint = ["D" + str(n).zfill(3) + str(s).zfill(3) + str(t).zfill(3) for t in grT for s in grSh for n in grN]

        VarNames = Cint + Dint
        nvar = self.opt_parameters['N']*self.opt_parameters['Sh']*self.opt_parameters['Sg'] + \
                self.opt_parameters['N']*self.opt_parameters['Sh']*self.opt_parameters['T']
        if nvar != len(VarNames):
            print(" ******  inconsistency with number of variables")

        # -------------------------------------------------------------------------
        #         objective function and bounds
        # ------------------------------------------------------------------------

        lbv = np.tile(0.0, nvar)  # lower bound 0 for all variables
        ubv = np.tile(1.0, nvar)  # upper bound 1 for all variables
        typev = "I" * nvar        # all variables are integer

        self.LCCOption[np.isnan(self.LCCOption)] = 0.0  # turn nans from investment costs to 0
        CostVec1a = [self.LCCOption[n,sh,sg]  for sg in grSg for sh in grSh for n in grN]  # investment costs
        # TODO add base costs
        #Sum the risk costs over time and sum with investment costs:
        CostVec1b = [np.sum(self.RiskGeotechnical[n,sg,:]) for sg in grSg for sh in grSh for n in grN]  #
        CostVec1 = list(np.add(CostVec1a,CostVec1b))


        CostVec2 = [self.RiskOverflow[n,sh,t]  for t in grT for sh in grSh for n in grN]  # risk costs of overflow
        CostVec = CostVec1 + CostVec2

        model.variables.add(obj=CostVec, lb=lbv, ub=ubv, types=typev, names=VarNames)
        self.CostVec = CostVec
        # -------------------------------------------------------------------------
        #         implement constraints
        # ------------------------------------------------------------------------

        # define lists that form the constraints
        A = list()  # A matrix of constraints
        b = list()  # b vector (right hand side) of equations

        # constraint XX: The initial condition Cint(n,s) = 0 for all n and s in N and S. This is not a valid constraint, it is an initial condition
        # constraint 1: There should be only 1 option implemented at each dike section: sum_s(Cint(s)=1 for all n in N
        C1 = list()
        for n in grN:
            slist = Cint_nd[n,:,:].ravel().tolist()
            nlist = [1.0] * (self.opt_parameters['Sg']*self.opt_parameters['Sh'])
            curconstraint = [slist,nlist]
            C1.append(curconstraint)
        A = A + C1
        senseV = "E"*len(C1)
        # b = b+[1.0]*self.opt_parameters['N']
        b = b+[1.0]*len(C1)

        print('constraint 1 implemented')

        # constraint 2: there is only 1 weakest section for overflow at any point in time
        C2 = list()
        for t in grT:
            # slist = Dint_nd[:,:,t].tolist()
            slist = [Dint_nd[n,s,t] for n in grN for s in grSh]
            nlist = [1.0] * (self.opt_parameters['N']*self.opt_parameters['Sh'])
            curconstraint = [slist,nlist]
            C2.append(curconstraint)
        A = A + C2
        senseV = senseV + "E"*len(C2)
        b = b+ [1.0]*len(C2)
        # b = b+ [1.0]*(self.opt_parameters['N']*self.opt_parameters['S'])

        print('constraint 2 implemented')

        #constraint 3: make sure that for overflow DY represents the weakest link
        # TODO Split this for different mechanisms!
        C3 = list()
        import sys
        for t in grT:
            for n in grN:
                for nst in grN:
                    for sst in grSh:
                        # derive the index of the relevant decision variables
                        index = self.Pf['Overflow'][n,:,t]>self.Pf['Overflow'][nst,sst,t]
                        index1 = np.where(index)[0]
                        ii = []
                        if np.size(index1) > 0:     #select the last. WHY THE LAST?
                            ii = index1

                        jj = {}
                        # jj = (self.opt_parameters['Sh'])*np.tile(1,self.opt_parameters['N'])
                        for kk in grN:

                            index = self.Pf['Overflow'][kk,:,t] <= self.Pf['Overflow'][nst,sst,t]
                            index1 = np.where(index)[0]
                            if np.size(index1) > 0:
                                jj[kk] = [index1]
                            else:
                                jj[kk] = []
                        slist = flatten([Cint_nd[n, sh, :].tolist() for sh in ii]) + \
                                flatten([Dint_nd[nh, sh, t].tolist() for nh in grN for sh in jj[nh]])
                        # print(len(slist))
                        nlist = [1.0]*len(slist)
                        curconstraint = [slist,nlist]
                        C3.append(curconstraint)
                        del curconstraint, slist,nlist
            print(str(sys.getsizeof(C3)) + ' bytes at t=' + str(t) + ' and n = ' + str(n))

        A = A + C3
        senseV = senseV + "L"*len(C3) # L means <=
        b = b+[1.0]*len(C3)

        print('constraint 3 implemented')

        print('binary constraints implemented in restriction of variables')

        # Add constraints to model:

        model.linear_constraints.add(lin_expr=A, senses=senseV, rhs=b)

        return model
    def readResults(self, Model,dir = None,MeasureTable=None):
        N = self.opt_parameters['N']
        Sh = self.opt_parameters['Sh']
        Sg = self.opt_parameters['Sg']
        T = self.opt_parameters['T']
        grN = range(N)
        grSh = range(Sh)
        grSg = range(Sg)
        grT = range(T)
        self.results = {}
        xs = Model['Values']
        ind = np.argwhere(xs)
        varnames = Model['Names']
        ones = np.array(varnames)[ind]

        LCCTotal = 0
        sections = ones[grN]
        # test = str(sections[i][0])
        measure = {}
        for i in grN:
            measure[np.int(str(sections[i][0])[1:4])] = [np.int(str(sections[i][0])[4:7]),np.int(str(sections[i][0])[
                                                                                                     7:])]
            # measure.append((np.int(str(sections[i][0])[1:4]), np.int(str(sections[i][0])[4:7])))

        LCCTotal = 0
        sectionnames = list(self.options.keys())
        sections = []
        measurenames = []
        yesno = []
        dcrest =[]
        dberm = []
        LCC = []
        ID = []
        ID2 = []
        for i in measure.keys():
            sections.append(sectionnames[i])
            print(sectionnames[i])
            if isinstance(MeasureTable,pd.DataFrame):
                if np.sum(measure[i]) != 0:
                    ID.append(self.options_geotechnical[sectionnames[i]].iloc[measure[i][1]-1]['ID'].values[0])
                    ID2.append(self.options_height[sectionnames[i]].iloc[measure[i][0] - 1]['ID'].values[0])
                    if ID[-1][-1] != ID2[-1]:
                        ID[-1] = ID[-1] + '+' + ID2[-1]
                        print(ID[-1])
                else:
                    ID.append('0')
                    ID2.append('0')
                    # MeasureTable.append(pd.DataFrame([['0', 'Do Nothing']],columns=['ID','Name']))

                if len(MeasureTable.loc[MeasureTable['ID'] == ID[-1]]) ==0:
                    if len(ID[-1])>1:
                        splitID = ID[-1].split('+')
                        newname = MeasureTable.loc[MeasureTable['ID'] == splitID[0]]['Name'].values + '+' + \
                                  MeasureTable.loc[MeasureTable['ID'] == splitID[1]]['Name'].values
                        newline = pd.DataFrame([[ID[-1], newname[0]]], columns=['ID', 'Name'])
                    else:
                        newline = pd.DataFrame([[ID[-1], 'Do Nothing']],columns=['ID','Name'])
                    MeasureTable = MeasureTable.append(newline)

                measurenames.append(MeasureTable.loc[MeasureTable['ID'] == ID[-1]]['Name'].values[0])
            else:
                if np.sum(measure[i]) != 0:
                    measurenames.append(self.options[sectionnames[i]].iloc[measure[i][1]-1]['type'].values[0])
                else:
                    measurenames.append('Do Nothing')
            if np.sum(measure[i]) != 0:
                yesno.append(self.options_geotechnical[sectionnames[i]].iloc[measure[i][1]-1]['yes/no'].values[0])
                dcrest.append(self.options_height[sectionnames[i]].iloc[measure[i][0]-1]['dcrest'].values[0])
                dberm.append(self.options_geotechnical[sectionnames[i]].iloc[measure[i][1]-1]['dberm'].values[0])

            else:
                yesno.append('no'); dcrest.append(0); dberm.append(0)

            LCC.append(self.LCCOption[i, measure[i][0], measure[i][1]])
            LCCTotal += self.LCCOption[i, measure[i][0], measure[i][1]]

        TakenMeasures = pd.DataFrame({'ID': ID, 'Section': sections,'LCC':LCC, 'name': measurenames, 'yes/no': yesno,
                                      'dcrest': dcrest, 'dberm': dberm})
        #add year
        self.results['measures'] = measure
        self.TakenMeasures = TakenMeasures
        data = pd.DataFrame({'Names': Model['Names'], 'Values': Model['Values'], 'Cost': self.CostVec})
        self.results['D_int'] = data.loc[data['Values'] == 1].iloc[-T:]
        self.results['C_int'] = data.loc[data['Values'] == 1].iloc[:-T]

        pd.set_option('display.max_columns', None)  # prevents trailing elipses
        # print(TakenMeasures)
        if dir:
            TakenMeasures.to_csv(dir.joinpath('TakenMeasures_MIP.csv'))
        else:
            TakenMeasures.to_csv('TakenMeasures_MIP.csv')
        ## reproduce objective:
        alldata = data.loc[data['Values'] == 1]
        self.results['TC'] = np.sum(alldata)['Cost']
        self.results['LCC+GeoRisk'] = np.sum(alldata.iloc[:-T])['Cost']
        self.results['OverflowRisk'] = np.sum(alldata.iloc[-T:])['Cost']


    def makeFinalSolution(self, path):
        pass
        self.FinalSolution = copy.deepcopy(TakenMeasures)

    def readResults_Fer(self, Model):
        N = self.opt_parameters['N']
        S = self.opt_parameters['S']
        T = self.opt_parameters['T']

        grN = range(N)
        grS = range(S)
        grT = range(T)

        # -------------------------------------------------------------------------
        #         read results
        # ------------------------------------------------------------------------

        TotalCost = Model.solution.get_objective_value()

        print("Solution status = ", Model.solution.status[Model.solution.get_status()])
        print("Solution value  = ", TotalCost)
        print("number of variables: " + "{0:0=2d}".format(Model.variables.get_num()))
        print("number of constraints: " + "{0:0=2d}".format(Model.linear_constraints.get_num()))

        # values of all decision variables
        xsol = Model.solution.get_values()

        # make nd-matrices
        solC_int = np.tile(np.nan, (N,S))
        solD_int = np.tile(np.nan, (N,S,T))
        count = 0
        for n in grN:
            for s in grS:
                solC_int[n,s] = xsol[count]
                count += 1
        for n in grN:
            for s in grS:
                for t in grT:
                    solD_int[n,s,t] = xsol[count]
                    count += 1

        self.results ={}
        self.results['C_int'] = solC_int
        self.results['D_int'] = solD_int

        # -------------------------------------------------------------------------
        #         verify cost function
        # ------------------------------------------------------------------------

        LCC = 0.0
        for n in grN:
            for s in grS:
                LCC = LCC + solC_int[n,s] *self.LCCOption[n,s]

        LCRisk = 0.0
        for n in grN:
            for s in grS:
                for t in grT:
                    LCRisk = LCRisk + solD_int[n,s,t] * self.Risk[n,s,t]

        TotalCostCheck = LCC +LCRisk
        Diff = TotalCostCheck - TotalCost
        PercError = Diff / TotalCostCheck
        if np.abs(PercError) < 1e-10:
            print("Cost function check succesful")
        else:
            print(" ******  Cost function check unsuccesful")
            print('LCC = ' + str(LCC))
            print('LCR = ' + str(LCRisk))
            print('TC  = ' + str(TotalCost))

    def checkConstraintSatisfaction(self, Model):
        N = self.opt_parameters['N']
        S = self.opt_parameters['S']
        T = self.opt_parameters['T']

        grN = range(N)
        grS = range(S)
        grT = range(T)
        # -------------------------------------------------------------------------
        #         verify if constraints are satisfied
        # ------------------------------------------------------------------------

        AllConstraintsSatisfied = True

        #C1
        GG = np.tile(1.0, N)
        for n in grN:
            GG[n] = np.sum(self.results['C_int'][n,:])

        if (GG==1).all():
            print('constraint 1 satisfied')
        else:
            print('Warning: constraint 1 not satisfied')
            AllConstraintsSatisfied = False
        #C2
        GG = np.tile(1.0, T)
        for t in range(0, T):
            GG[t] = sum(sum(self.results['D_int'][:, :, t]))

        if (GG == 1).all():
            print("constraint C2 satisfied")
        else:
            print(" ******  warning: constraint C2 not satisfied")
            AllConstraintsSatisfied = False

            #C3
        pass

    def plotBetaTime(self,Traject, typ='single',fig_id = None,path = None, horizon = 100):
        step = 0
        beta_t = []
        plt.figure(100)
        for i in self.Probabilities:
            step += 1
            beta_t0, p = calcTrajectProb(i, horizon=horizon)
            beta_t.append(beta_t0)
            t = range(2025,2025+horizon)
            plt.plot(t, beta_t0, label=self.type+ ' stap ' + str(step))
        if typ == 'single':
            plt.plot([2025, 2025+horizon], [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])],
                     'k--', label='Norm')
            plt.xlabel('Time')
            plt.ylabel(r'$\beta$')
            plt.legend()
            plt.savefig(path.joinpath('figures', 'BetaInTime' + self.type + '.png'), bbox_inches='tight')
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
            symbols = ['*', 'o', '^', 's', 'p', 'X', 'd', 'h', '>', '.', '<', 'v', '3', 'P', 'D']
            MeasureTable = MeasureTable.assign(symbol=symbols[0: len(MeasureTable)])

        if symbolmode == 'off':
            symbols = None

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
            for i in range(self.TakenMeasures['Section'].size):
                if not np.isnan(self.TakenMeasures['LCC'].iloc[i]):
                    x += self.TakenMeasures['LCC'].iloc[i]
                else:
                    pass
                Costs.append(x)

        elif cost_type == 'Initial':
            for i in range(self.TakenMeasures['Section'].size):
                if i > 0:
                    years = self.options[self.TakenMeasures.iloc[i]['Section']].iloc[self.TakenMeasures.iloc[i]['option_index']]['year'].values[0]
                    if not isinstance(years, int):
                        for ij in range(len(years)):
                            if years[ij] == 0:
                                x += self.options[self.TakenMeasures.iloc[i]['Section']].iloc[self.TakenMeasures.iloc[i]['option_index']]['cost'].values[0][ij]

                    elif isinstance(years, int):
                        if years > 0:
                            pass
                        else:
                            x += self.TakenMeasures['LCC'].iloc[i]

                    Costs.append(x)
                else:
                    Costs.append(x)

        Costs = np.divide(Costs, 1e6)
        if typ == 'single':
            plt.figure(101)
            if symbolmode == 'off':
                plt.plot(Costs, betas, 'or-', label=self.type)

            if symbolmode == 'on':
                if beta_or_prob == 'beta':
                    plt.plot(Costs, betas, 'r-', label=self.type)

                if beta_or_prob == 'prob':
                    plt.plot(Costs, pfs, 'r-', label=self.type)

                print('add symbols for single plot')        # TO DO

            plt.plot([0, np.max(Costs)], [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])], 'k--', label='Norm')
            plt.xlabel('Total LCC in M€')
            if beta_or_prob == 'beta':
                plt.ylabel(r'$\beta$')
                plt.title('Total LCC versus ' + r'$\beta$' + ' in year ' + str(t + 2025))
                plt.savefig(path.joinpath('figures', 'BetavsCosts' + self.type + '_' + str(t + 2025) + '.png'), bbox_inches='tight')

            if beta_or_prob == 'prob':
                plt.ylabel(r'Failure probability $P_f$')
                plt.title('Total LCC versus ' + r'$P_f$' + ' in year ' + str(t + 2025))
                plt.savefig(path.joinpath('figures', 'PfvsCosts' + self.type + '_' + str(t + 2025) + '.png'), bbox_inches='tight')

            plt.close(101)
        else:
            plt.figure(fig_id)
            if beta_or_prob == 'beta':
                if labels != None:
                    if symbolmode == 'off':
                        plt.plot(Costs, betas, 'o-', label=labels, color=linecolor, linestyle=linestyle)

                    if symbolmode == 'on':
                        plt.plot(Costs, betas, '-', label=labels, color=linecolor, linestyle=linestyle, zorder=1)

                else:
                    if symbolmode == 'off':
                        plt.plot(Costs, betas, 'o-', label=self.type, color=linecolor, linestyle=linestyle)

                    if symbolmode == 'on':
                        plt.plot(Costs, betas, '-', label=self.type, color=linecolor, linestyle=linestyle, zorder=1)

            elif beta_or_prob == 'prob':
                if labels != None:
                    if symbolmode == 'off':
                        plt.plot(Costs, pfs, 'o-', label=labels, color=linecolor, linestyle=linestyle)

                    if symbolmode == 'on':
                        plt.plot(Costs, pfs, '-', label=labels, color=linecolor, linestyle=linestyle, zorder=1)
                else:
                    if symbolmode == 'off':
                        plt.plot(Costs, pfs, 'o-', label=self.type, color=linecolor, linestyle=linestyle)

                    if symbolmode == 'on':
                        plt.plot(Costs, pfs, '-', label=self.type, color=linecolor, linestyle=linestyle, zorder=1)

            if symbolmode == 'on':
                if beta_or_prob == 'beta':
                    interval = .07
                    base = np.max(betas) + interval
                    ycoord = np.array([base, base + interval, base + 2 * interval, base + 3 * interval])
                    ycoords = np.tile(ycoord, np.int(np.ceil(len(Costs) / len(ycoord))))

                elif beta_or_prob == 'prob':
                    interval = 2
                    base = np.min(pfs)/interval
                    ycoord = np.array([base, base/interval, base/(2 * interval), base/(3 * interval)])
                    ycoords = np.tile(ycoord, np.int(np.ceil(len(Costs) / len(ycoord))))

                for i in range(len(Costs)):
                    line = self.TakenMeasures.iloc[i]
                    if line['option_index'] != None:
                        if isinstance(line['ID'], list):line['ID'] = '+'.join(line['ID'])
                        if Costs[i] > Costs[i-1]:
                            if beta_or_prob == 'beta':
                                plt.scatter(Costs[i],betas[i],marker=MeasureTable.loc[MeasureTable['ID']==line['ID']]['symbol'].values[0],label=MeasureTable.loc[MeasureTable['ID']==line['ID']]['Name'].values[0],color=linecolor,edgecolors='k',linewidths=.5,zorder=2)
                                plt.vlines(Costs[i],betas[i]+.05,ycoords[i]-.05,colors ='tab:gray', linestyles =':',zorder = 1)
                            elif beta_or_prob == 'prob':
                                plt.scatter(Costs[i], pfs[i], marker=MeasureTable.loc[MeasureTable['ID'] == line['ID']]['symbol'].values[0], label=MeasureTable.loc[MeasureTable['ID'] == line['ID']]['Name'].values[0], color=linecolor, edgecolors='k', linewidths=.5, zorder=2)
                                plt.vlines(Costs[i], pfs[i], ycoords[i], colors='tab:gray', linestyles=':', zorder=1)

                        plt.text(Costs[i], ycoords[i], line['Section'][-2:], fontdict={'size': 8}, color=linecolor, horizontalalignment='center', zorder=3)

            if labelmeasures == 'on':
                for i in range(len(Costs)):
                    line = self.TakenMeasures.iloc[i]
                    if isinstance(line['option_index'], int):
                        meastyp = self.options[line['Section']].iloc[line['option_index']]['type'].values[0]
                        if isinstance(meastyp, list):
                            meastyp = str(meastyp[0] + '+' + meastyp[1])

                    if i > 0:
                        if beta_or_prob == 'beta':
                            plt.text(Costs[i], betas[i] - .1, line['Section'] + ': ' + meastyp)

                        if beta_or_prob == 'prob':
                            plt.text(Costs[i], pfs[i] / 2, line['Section'] + ': ' + meastyp)

            if last == 'yes':
                axes = plt.gca()
                xmax = np.max([axes.get_xlim()[1], np.max(Costs)])
                ceiling = np.ceil(np.max([xmax, np.max(Costs)]) / 10) * 10
                if beta_or_prob == 'beta':
                    plt.plot([0, ceiling], [-norm.ppf(Traject.GeneralInfo['Pmax']), -norm.ppf(Traject.GeneralInfo['Pmax'])], 'k--', label='Norm')

                if beta_or_prob == 'prob':
                    plt.plot([0, ceiling], [Traject.GeneralInfo['Pmax'], Traject.GeneralInfo['Pmax']], 'k--', label='Norm')

                if cost_type == 'LCC':
                    plt.xlabel('Total LCC in M€')

                if cost_type == 'Initial':
                    plt.xlabel('Cost in 2025 in M€')

                if beta_or_prob == 'beta':
                    plt.ylabel(r'$\beta$')

                if beta_or_prob == 'prob':
                    plt.ylabel(r'$P_f$')

                if beta_or_prob == 'prob':
                    axes.set_yscale('log')
                    axes.invert_yaxis()

                plt.xticks(np.arange(0, ceiling + 1, 10))
                axes.set_xlim(left=0, right=ceiling)
                plt.grid()

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc=4, fontsize='x-small')

                # plt.legend(loc=5)

                if cost_type == 'LCC':
                    if beta_or_prob == 'beta':
                        plt.title('Total LCC versus ' + r'$\beta$' + ' in year ' + str(t + 2025))

                    if beta_or_prob == 'prob':
                        plt.title('Total LCC versus ' + r'$P_f$' + ' in year ' + str(t + 2025))

                if cost_type == 'Initial':
                    if beta_or_prob == 'beta':
                        plt.title('Costs versus ' + r'$\beta$' + ' in year ' + str(t + 2025))

                    if beta_or_prob == 'prob':
                        plt.title('Costs versus ' + r'$P_f$' + ' in year ' + str(t + 2025))

                if beta_or_prob == 'beta':
                    plt.title(r'Relation between $\beta$ and investment costs in M€')

                if beta_or_prob == 'prob':
                    plt.title(r'Relation between $P_f$ and investment costs in M€')

                plt.xlabel('Investment costs in M€')

                if path is None:
                    plt.show()
                else:
                    if name is None:
                        if beta_or_prob == 'beta':
                            if cost_type == 'LCC':
                                plt.savefig(path.joinpath('BetavsLCC_t' + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)

                            if cost_type == 'Initial':
                                plt.savefig(path.joinpath('BetavsCosts_t' + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)

                        elif beta_or_prob == 'prob':
                            if cost_type == 'LCC':
                                plt.savefig(path.joinpath('PfvsLCC_t' + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)

                            if cost_type == 'Initial':
                                plt.savefig(path.joinpath('PfvsCosts_t' + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)
                    else:
                        if cost_type == 'LCC':
                            plt.savefig(path.joinpath('LCC_' + name + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)

                        if cost_type == 'Initial':
                            plt.savefig(path.joinpath('Cost_' + name + str(t + 2025) + '.png'), bbox_inches='tight', dpi=300)

                    plt.close(fig_id)

        if outputcsv:
            data = np.array([Costs.T, np.array(betas)]).T
            data = pd.DataFrame(data, columns=['Cost', 'beta'])
            if cost_type == 'LCC':
                data.to_csv(path.joinpath('BetavsLCC' + thislabel + '_t' + str(t+2025) + '.csv'))

            if cost_type == 'Initial':
                data.to_csv(path.joinpath('BetavsCost' + thislabel + '_t' + str(t + 2025) + '.csv'))
            pass

    def plotInvestmentSteps(self, TestCase, path=None, figure_size=(6, 4)):
        updatedAssessment = copy.deepcopy(TestCase)
        for i in range(1, len(self.TakenMeasures)):
            if i == 6:
                print()
                pass

            updatedAssessment.plotReliabilityofDikeTraject(first=True, last=False, alpha=0.3, fig_size=figure_size)
            updatedAssessment.updateProbabilities(self.Probabilities[i], self.TakenMeasures.iloc[i]["Section"])
            updatedAssessment.plotReliabilityofDikeTraject(pathname=path, first=False, last=True, type=i, fig_size=figure_size)

        # for i in self.TakenMeasures:

    def writeProbabilitiesCSV(self, path, type):
        # with open(path + '\\ReliabilityLog_' + type + '.csv', 'w') as f:
        for i in range(len(self.Probabilities)):
            name = path.joinpath('ReliabilityLog_' + type + '_Step' + str(i) + '.csv')
            # measurerow = self.TakenMeasures.iloc[i]['Section'] + ',' + self.TakenMeasures.iloc[i]['name'] + ',' + str(self.TakenMeasures.iloc[i]['params'])+ ',' + str(self.TakenMeasures.iloc[i]['LCC'])
            # f.write(measurerow)

            # self.TakenMeasures.iloc[i].to_csv(f, header=True)
            self.Probabilities[i].to_csv(path_or_buf=name, header=True)
    def determineRiskCostCurve(self,TrajectObject,PATH):
        if PATH:
            PATH.mkdir(parents=True, exist_ok=True)
        else:
            PATH = False
        if not hasattr(self, 'TakenMeasures'):
            raise TypeError('TakenMeasures not found')
        TR = []
        LCC = []
        if self.type == 'Heuristic': #do a loop

            LCC = np.cumsum(self.TakenMeasures['LCC'].values)
            count = 0
            for i in self.Probabilities:
                count += 1
                TR.append(calcLifeCycleRisks(i, self.r, np.max(TrajectObject.GeneralInfo['T']),
                                             TrajectObject.GeneralInfo['FloodDamage'],dumpPt=PATH.joinpath(
                        'Heuristic_step_' +str(
                    count) + '.csv')))
        elif self.type == 'MixedInteger':
            LCC = np.sum(self.TakenMeasures['LCC'].values)
            #find the ids of the options
            section = []
            option_index = []
            for i in self.TakenMeasures.iterrows():
                data = i[1]
                section.append(data['Section'])
                if data['name'] != 'Do Nothing':
                    option_index.append(self.options[data['Section']].loc[
                                            (self.options[data['Section']]['ID'] == data['ID']) &
                                            (self.options[data['Section']]['yes/no'].values == data['yes/no']) &
                                            (self.options[data['Section']]['dcrest'] == data['dcrest']) &
                                            (self.options[data['Section']]['dberm'] == data['dberm'])].index.values[0])
                else:
                    option_index.append(-999)
            #implement the options 1 by 1
            Probability = makeTrajectDF(TrajectObject,TrajectObject.GeneralInfo['T'])
            ProbabilitySteps = []
            ProbabilitySteps.append(copy.deepcopy(Probability))
            for i in range(0,len(option_index)):
                if option_index[i] > -999:
                    Probability = ImplementOption(section[i], Probability, self.options[section[i]].iloc[
                        option_index[i]])
                ProbabilitySteps.append(copy.deepcopy(Probability))

            #evaluate the risk after the last step
            TR = calcLifeCycleRisks(ProbabilitySteps[-1],self.r, np.max(TrajectObject.GeneralInfo['T']),
                                             TrajectObject.GeneralInfo['FloodDamage'],dumpPt=PATH.joinpath(
                    'MixedInteger.csv'))

            #alternative (should yield the same). Small differences for now due to difference in Risk computation:
            # RiskOverflow = np.empty((np.shape(self.RiskGeotechnical)[2], len(self.results['measures'])))
            # RiskGeotechnical = np.empty((np.shape(self.RiskGeotechnical)[2], len(self.results['measures'])))
            # for i in self.results['measures'].keys():
            #     indices = self.results['measures'][i]
            #     print(indices)
            #     # print(self.results['measures'][i])
            #     RiskOverflow[:, i] = self.RiskOverflow[i, indices[0], :]
            #     RiskGeotechnical[:, i] = self.RiskGeotechnical[i, indices[1], :]
            #
            # Risk = np.sum(np.max(RiskOverflow, axis=1)) + np.sum(RiskGeotechnical)

        return TR, LCC
#This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def DetermineNewGeometry(geometry_change, direction, initial,bermheight = 2,geometry_plot = False,plot_dir = None, slope_in = False):


    if len(initial) == 6:
        bermheight = initial.iloc[2]['z']
    elif len(initial) == 4:
        pass
    #Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm  = geometry_change[1]
    if dcrest > 1.01 and dberm > 31 and geometry_plot:
        geometry_plot = False
    geometry = initial.values
    cur_crest = np.max(geometry[:,1])
    new_crest = cur_crest+dcrest
    if geometry_plot:
        plt.plot(geometry[:, 0], geometry[:, 1],'k')
    if direction == 'outward':
        print("WARNING: outward reinforcement is NOT UP TO DATE!!!!")
        new_geometry = copy.deepcopy(geometry)
        for i in range(len(new_geometry)):
            #Run over points from the outside.
            if i > 0:
                slope = (geometry[i - 1][1] - geometry[i][1]) / (geometry[i - 1][0] - geometry[i][0])
                if slope > 0 and new_geometry[i, 1] == cur_crest:  # outer slope
                    new_geometry[i][0] = (new_geometry[i][0] + dcrest / np.abs(slope))
                    new_geometry[i][1] = new_crest
                elif slope == 0:  # This is the crest
                    new_geometry[i][0] = new_geometry[i - 1][0] + np.abs(geometry[i - 1][0] - geometry[i][0])
                    new_geometry[i][1] = new_crest
                elif slope < 0 and new_geometry[i, 1] != cur_crest:  # This is the inner slope
                    new_geometry[i][0] = new_geometry[i - 1][0] + (new_geometry[i - 1][1] - new_geometry[i][1]) / np.abs(slope)
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
            for i in range(len(new_geometry)):
                new_geometry[i][0] -= dberm

        if geometry_plot:
            plt.plot(new_geometry[:, 0], new_geometry[:, 1], '--r')

    elif direction == 'inward':
        new_geometry = copy.deepcopy(geometry)
        # we start at the outer toe so reverse:

        for i in reversed(range(len(new_geometry)-1)):
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
                if slope_in:
                    # print('check geometry formula, NOT CORRECTED YET')
                    new_geometry[i][0] = new_geometry[i + 1][0] - (new_geometry[i + 1][1] - new_geometry[i][1]) * slope_in
                else:
                    new_geometry[i][0] = new_geometry[i + 1][0] + (new_geometry[i][1] - new_geometry[i + 1][
                        1])/np.abs(slope)
                # new_geometry[i-1][0] = new_geometry[i][0] - (new_geometry[i][1] - new_geometry[i-1][1]) * slope_in

        if dberm > 0 and len(geometry) == 4:     #add a berm if the points dont exist yet
            if slope_in:
                x1 = new_geometry[0][0] + (bermheight*slope_in)
            else:
                x1 = new_geometry[0][0] + (bermheight/np.abs(slope))
            x2 = x1 - dberm
            y = bermheight + new_geometry[0][1]
            new_geometry[0][0] = new_geometry[0][0] - dberm
            new_geometry = np.insert(new_geometry, [1], np.array([x1, y]), axis=0)
            new_geometry = np.insert(new_geometry, [1], np.array([x2, y]), axis=0)
        elif dberm > 0 and len(geometry) == 6:
            pass

    #calculate the area difference
    # geometry = np.append(geometry, np.array([new_geometry[-1]]), axis=0)
    area_old, polygon_old = calculateArea(geometry)
    area_new, polygon_new = calculateArea(new_geometry)
    poly_diff = polygon_new.difference(polygon_old)
    area_difference = poly_diff.area
    if geometry_plot:
        if hasattr(poly_diff, 'geoms'):
            for i in range(len(poly_diff.geoms)):
                x1, y1 = poly_diff[i].exterior.xy
                plt.fill(x1, y1, 'r--')

        else:
            x1, y1 = poly_diff.exterior.xy
            plt.fill(x1, y1, 'r--')

        plt.text(np.mean(new_geometry[:, 0]), np.max(new_geometry[:, 1]), 'Area difference = ' + '{:.4}'.format(str(area_difference)) + ' $m^2$')
        if plot_dir == None:
            plt.show()
        else:
            plt.savefig(plot_dir.joinpath('Geometry_' + str(dberm) + '_' + str(dcrest) + '.png'))
            plt.close()

    return new_geometry, area_difference

# script to calculate the area difference of a new geometry after reinforcement (compared to old one)
def calculateArea(geometry):
    extra = np.empty((1, 2))
    if geometry[-1][1] > geometry[0][1]:
        extra[0, 0] = geometry[-1][0]
        extra[0, 1] = geometry[0][1]
        geometry = np.append(geometry, np.array(extra), axis=0)
    elif geometry[-1][1] < geometry[0][1]:
        extra[0, 0] = geometry[0][0]; extra[0, 1] = geometry[-1][1]
        geometry = np.insert(geometry, [0], np.array(extra), axis=0)

    bottomlevel = np.min(geometry[:, 1])
    area = 0

    for i in range(1, len(geometry)):
        a = np.abs(geometry[i-1][0] - geometry[i][0]) * (0.5 * np.abs(geometry[i - 1][1]-geometry[i][1]) + 1.0 * (np.min((geometry[i-1][1], geometry[i][1])) - bottomlevel))
        area += a

    polypoints= []

    for i in range(len(geometry)):
        polypoints.append((geometry[i, 0], geometry[i, 1]))
    polygon = Polygon(polypoints)
    return area, polygon

#Script to determine the costs of a reinforcement:
def DetermineCosts(parameters, type, length, reinf_pars = None, housing = None, area_difference = None):
    if type == 'Soil reinforcement':
        if parameters['StabilityScreen'] == 'no':
            C = parameters['C_start'] + area_difference*parameters['C_unit'] * length
            if isinstance(housing, pd.DataFrame) and reinf_pars[1] > 0.:
                C += parameters['C_house'] * housing.loc[float(reinf_pars[1])]['cumulative']

        elif parameters['StabilityScreen'] == 'yes':
            C = parameters['C_start'] + area_difference*parameters['C_unit'] * length + parameters['C_unit2'] * parameters['Depth'] * length
            if isinstance(housing, pd.DataFrame) and reinf_pars[1] > 0.:
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

#Script to determine the required crest height for a certain year
def ProbabilisticDesign(design_variable, strength_input, Pt, horizon = 50, loadchange = 0, mechanism='Overflow'):
    if mechanism == 'Overflow':
        #determine the crest required for the target
        h_crest, beta = OverflowSimple(strength_input['h_crest'], strength_input['q_crest'], strength_input['h_c'], strength_input['q_c'], strength_input['beta'], mode='design', Pt=Pt, design_variable=design_variable)
        #add temporal changes due to settlement and climate change
        h_crest = h_crest + horizon * (strength_input['dhc(t)'] + loadchange)
        return h_crest

# This script combines two sets of measures to a single option
def MeasureCombinations(combinables, partials, solutions,splitparams = False):
    CombinedMeasures = pd.DataFrame(columns=combinables.columns)

    #loop over partials
    for i, row1 in partials.iterrows():
    #combine with all combinables
        for j, row2 in combinables.iterrows():
            ID = '+'.join((row1['ID'].values[0], row2['ID'].values[0]))
            types = [row1['type'].values[0], row2['type'].values[0]]
            year = [row1['year'].values[0], row2['year'].values[0]]
            if splitparams:
                params = [row1['yes/no'].values[0], row2['dcrest'].values[0],row2['dberm'].values[0]]
            else:
                params = [row1['params'].values[0], row2['params'].values[0]]
            Cost = [row1['cost'].values[0], row2['cost'].values[0]]
            #combine betas
            #take maximums of mechanisms except if it is about StabilityInner for partial Stability Screen
            betas = []
            years = []

            for ij in partials.columns:
                if ij[0] != 'Section' and ij[1] != '':     #It is a beta value
                    beta = np.maximum(row1[ij], row2[ij])
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

            if splitparams:
                in1 = [ID, types, 'combined', year, params[0],params[1],params[2], Cost]
            else:
                in1 = [ID, types, 'combined', year, params, Cost]

            allin = pd.DataFrame([in1 + betas], columns=combinables.columns)
            CombinedMeasures = CombinedMeasures.append(allin)
    return CombinedMeasures

def makeTrajectDF(traject, cols):
    # cols = cols[1:]
    sections = []

    for i in traject.Sections:
        sections.append(i.name)

    mechanisms = list(traject.Sections[0].MechanismData.keys()) + ['Section']
    df_index = pd.MultiIndex.from_product([sections, mechanisms], names=['name', 'mechanism'])
    TrajectProbability = pd.DataFrame(columns=cols, index=df_index)

    for i in traject.Sections:
        for j in mechanisms:
            TrajectProbability.loc[(i.name, j)] = list(i.Reliability.SectionReliability.loc[j])

    return TrajectProbability

#hereafter a bunch of functions to compute costs, risks and probabilities over time are defined:
def calcTC(section_options, r=0.03, horizon=100):
    costs = section_options['cost'].values
    years = section_options['year'].values
    discountfactors = list(map(lambda x: 1 / (1 + r) ** np.array(x), years))
    TC = list(map(lambda c, r: c * r, costs, discountfactors))
    return np.array(list(map(lambda c: np.sum(c), TC)))

def calcTR(section, section_options, base_traject, original_section, r=0.03, horizon=100, damage=1e9):
    #section: the section name
    #section_options: all options for the section
    #base_traject: traject probability with all implemented measures
    #takenmeasures: object with all measures taken
    #original section: series of probabilities of section, before taking a measure.
    if damage == 1e9:
        print('WARNING NO DAMAGE DEFINED')

    TotalRisk = []
    dR = []
    mechs = np.unique(base_traject.index.get_level_values('mechanism').values)
    sections = np.unique(base_traject.index.get_level_values('name').values)
    section_idx = np.where(sections == section)[0]
    section_options_array = {}
    base_array = {}
    TotalRisk = []
    dR = []

    for i in mechs:
        base_array[i] = base_traject.xs(i, level=1).values.astype('float')
        if isinstance(section_options, pd.DataFrame):
            section_options_array[i] = section_options.xs(i, level=0, axis=1).values.astype('float')
            range_idx = len(section_options_array[mechs[0]])

        if isinstance(section_options, pd.Series):
            section_options_array[i] = section_options.xs(i, level=0).values.astype('float')
            range_idx = 0

    if 'section_options_array' in locals():
        base_risk = calcLifeCycleRisks(base_array, r, horizon, damage, datatype='Array', ts=base_traject.columns.values, mechs=mechs)

        for i in range(range_idx):
                TR = calcLifeCycleRisks(base_array, r, horizon, damage, change=section_options_array, section=section_idx, datatype='Array', ts=base_traject.columns.values, mechs=mechs, option=i)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)
    else:
        base_risk = calcLifeCycleRisks(base_traject, r, horizon, damage)
        if isinstance(section_options, pd.DataFrame):
            for i, row in section_options.iterrows():
                TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change=row, section=section)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)

        elif isinstance(section_options, pd.Series):
            TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change=section_options, section=section)
            TotalRisk.append(TR)
            dR.append(base_risk - TR)

    return base_risk, dR, TotalRisk

def calcLifeCycleRisks(base0, r, horizon,damage, change=None, section=None, datatype='DataFrame', ts=None,mechs=False,
                       option=None,dumpPt=False):
    base = copy.deepcopy(base0)
    if datatype == 'DataFrame':
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        if isinstance(change, pd.Series):
            for i in mechs:
                #This is not very efficient. Could be improved.
                base.loc[(section, i)] = change.loc[i]
        else:
            pass

        beta_t, p_t = calcTrajectProb(base, horizon=horizon)
    elif datatype == 'Array':
        if isinstance(change, dict):
            for i in mechs:
                base[i][section] = change[i][option]
        else:
            pass
        if not (isinstance(ts,np.ndarray) or isinstance(ts,list)):
            ts = np.array(range(0,horizon))
        if not isinstance(mechs,np.ndarray): mechs = np.array(list(base.keys()))
        beta_t, p_t = calcTrajectProb(base, horizon=horizon, datatype='Arrays', ts=ts, mechs=mechs)

    # trange = np.arange(0, horizon + 1, 1)
    trange = np.arange(0, horizon, 1)
    D_t = damage / (1 + r) ** trange
    risk_t = p_t * D_t
    if dumpPt:
        np.savetxt(dumpPt,p_t,delimiter=",")
    TR = np.sum(risk_t)
    return TR

def calcTrajectProb(base, horizon=None, datatype='DataFrame', ts=None, mechs=False):
    pfs = {}
    trange = np.arange(0, horizon, 1)
    if datatype == 'DataFrame':
        ts = base.columns.values
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        # mechs = ['Overflow']
    # pf_traject = np.zeros((len(ts),))
    pf_traject = np.zeros((len(trange),))

    for i in mechs:
        if i != 'Section':
            if datatype == 'DataFrame':
                betas = base.xs(i, level='mechanism').values.astype('float')
            else:
                betas = base[i]
            beta_interp = interp1d(ts,betas)
            pfs[i] = norm.cdf(-beta_interp(trange))
            # pfs[i] = norm.cdf(-betas)
            pnonfs = 1 - pfs[i]
            if i == 'Overflow':
                pf_traject += np.max(pfs[i], axis=0)
            else:
                pf_traject += np.sum(pfs[i], axis=0)
                # pf_traject += 1-np.prod(pnonfs, axis=0)

    # elif datatype == 'Arrays':
    #     pf_traject = np.zeros((len(ts),))
    #     # TODO refactor this routine so it is more logical
    #     for i in mechs:
    #         if i != 'Section':
    #             pfs[i] = norm.cdf(-base[i])
    #             pnonfs = 1 - pfs[i]
    #             if i == 'Overflow':
    #                 pf_traject += np.max(pfs[i], axis=0)
    #             else:
    #                 # pf_traject += np.sum(pfs[i], axis=0)
    #                 pf_traject += 1-np.prod(pnonfs, axis=0)
    #                 # print('old:' + str(np.sum(pfs[i], axis=0)))
    #                 # print('new:' + str(1-np.prod(pnonfs, axis = 0)))

    # trange = np.arange(0, horizon + 1, 1)

    ## INTERPOLATION AFTER COMBINATION:
    # pfail = interp1d(ts,pf_traject)
    # p_t1 = norm.cdf(-pfail(trange))
    # betafail = interp1d(ts, -norm.ppf(pf_traject),kind='linear')
    # beta_t = betafail(trange)
    # p_t = norm.cdf(-np.array(beta_t, dtype=np.float64))

    beta_t = -norm.ppf(pf_traject)
    p_t = pf_traject
    return beta_t, p_t

#this function changes the trajectprobability of a measure is implemented:
def ImplementOption(section, TrajectProbability, newProbability):
    mechs = np.unique(TrajectProbability.index.get_level_values('mechanism').values)
    #change trajectprobability by changing probability for each mechanism
    for i in mechs:
        TrajectProbability.loc[(section, i)] = newProbability[i]
    return TrajectProbability

def split_options(options):
    options_height = copy.deepcopy(options)
    options_geotechnical = copy.deepcopy(options)
    for i in options:
        #filter all different measures for height
        options_height[i] = options_height[i].loc[options[i]['class'] != 'combined']
        options_height[i] = options_height[i].loc[(options[i]['type'] == 'Diaphragm Wall') | (options[i]['dberm'] == 0)]




        #now we filter all geotechnical measures
        #first all crest heights are thrown out
        options_geotechnical[i] = options_geotechnical[i].loc[
            (options_geotechnical[i]['dcrest'] == 0.0) | (options_geotechnical[i]['dcrest']==-999) |
            ((options_geotechnical[i]['class'] == 'combined') & (options_geotechnical[i]['dberm'] == 0))]

        #subtract startcosts, only for height.
        startcosts = np.min(options_height[i][(options_height[i]['type'] == 'Soil reinforcement')]['cost'])

        options_height[i]['cost'] = np.where(options_height[i]['type'] == 'Soil reinforcement',
                                             np.subtract(options_height[i]['cost'], startcosts),
                                             options_height[i]['cost'])
        options_geotechnical[i] = options_geotechnical[i].reset_index(drop=True)
        options_height[i]       = options_height[i].reset_index(drop=True)

        #loop for the geotechnical stuff:
        newcosts = []
        for ij in options_geotechnical[i].index:
            if options_geotechnical[i].iloc[ij]['type'].values[0] == 'Soil reinforcement':
                newcosts.append(options_geotechnical[i].iloc[ij]['cost'].values[0])
            elif options_geotechnical[i].iloc[ij]['class'].values[0] == 'combined':
                newcosts.append([options_geotechnical[i].iloc[ij]['cost'].values[0][0],
                                 options_geotechnical[i].iloc[ij]['cost'].values[0][1]])
            else:
                newcosts.append(options_geotechnical[i].iloc[ij]['cost'].values[0])
        options_geotechnical[i]['cost'] = newcosts
        #only keep reliability of relevant mechanisms in dictionary
        options_height[i].drop(['Piping','StabilityInner','Section'],axis=1)
        options_geotechnical[i].drop(['Overflow','Section'],axis=1)
    return options_height,options_geotechnical

def SolveMIP(MIPModel):

    MixedIntegerSolution = MIPModel.solve()
    return MixedIntegerSolution