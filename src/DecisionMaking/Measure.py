import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.Mechanisms import OverflowHRING, OverflowSimple
from src.FloodDefenceSystem.ReliabilityCalculation import (
    MechanismReliabilityCollection,
    beta_SF_StabilityInner,
)
from src.FloodDefenceSystem.SectionReliability import SectionReliability

"""Important: a measure is a single type of reinforcement, so for instance a stability screen. A solution can be a COMBINATION of measures (e.g. a stability screen with a berm)"""
class Measure():
    """Possible change: create subclasses for different measures to make the below code more neat. Can be done jointly with adding outward reinforcement"""
    #class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    def __init__(self, inputs, config:VrtoolConfig):
        self.parameters = {}
        for i in range(0,len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]
        
        self.crest_step = config.crest_step
        self.berm_step = config.berm_step
        self.input_directory = config.input_directory
        self.t_0 = config.t_0
        self.geometry_plot = config.geometry_plot
        self.unit_costs = config.unit_costs

    def evaluateMeasure(self,DikeSection,TrajectInfo,preserve_slope = False):
        raise Exception('define subclass of measure')
    #     from HelperFunctions import createDir
    #
    #     #To be added: year property to distinguish the same measure in year 2025 and 2045
    #     type = self.parameters['Type']
    #     mechanisms = DikeSection.Reliability.Mechanisms.keys()
    #     SFincrease = 0.2        #for stability screen
    #     if config.geometry_plot:
    #         plt.figure(1000) 
    #         createDir(config.directory.joinpath('figures',DikeSection.name,'Geometry'))

        #different types of measures:
class SoilReinforcement(Measure):
    # type == 'Soil reinforcement':
    def evaluateMeasure(self, DikeSection, TrajectInfo, plot_dir = False, preserve_slope=False):
    # def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        # Measure.__init__(self,inputs)
        # self. parameters = measure.parameters

        SFincrease = 0.2  # for stability screen

        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        crest_step = self.crest_step
        berm_step = self.berm_step
        crestrange = np.linspace(self.parameters['dcrest_min'], self.parameters['dcrest_max'], np.int(1 + (self.parameters['dcrest_max']-self.parameters['dcrest_min']) / crest_step))
        #TODO: CLEAN UP, make distinction between inwards and outwards, so xin, xout and y,and adapt DetermineNewGeometry
        if self.parameters['Direction'] == 'outward':
            if np.size(berm_step)>1:
                max_berm = self.parameters['max_outward']+self.parameters['max_inward']
                bermrange = berm_step[:len(np.where((berm_step <= max_berm))[0])]
            else:
                bermrange = np.linspace(0., self.parameters['max_outward'], np.int(1+(self.parameters['max_outward']/berm_step)))
        elif self.parameters['Direction'] == 'inward':
            if np.size(berm_step)>1:
                max_berm = self.parameters['max_inward']
                bermrange = berm_step[:len(np.where((berm_step <= max_berm))[0])]
            else:
                bermrange = np.linspace(0., self.parameters['max_inward'], np.int(1+(self.parameters['max_inward']/berm_step)))
        else:
            raise Exception('unkown direction')

        measures = [[x,y] for x in crestrange for y in bermrange]
        if not preserve_slope:
            slope_in = 4
            slope_out = 3 #inner and outer slope
        else:
            slope_in = False
            slope_out = False

        self.measures = []
        if self.parameters['StabilityScreen'] == 'yes':
            if 'd_cover' in DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input:
                self.parameters['Depth'] = np.max([DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability[
                                               '0'].Input.input['d_cover'] + 1., 8.])
            else:
                self.parameters['Depth'] = 6. #TODO: implement a better depth estimate based on d_cover

        for j in measures:
            if self.parameters['Direction'] == 'outward':
                k = max(0, j[1]-self.parameters['max_inward']) #correction max_outward
            else:
                k = j[1]
            self.measures.append({})
            self.measures[-1]['dcrest'] =j[0]
            self.measures[-1]['dberm'] = j[1]
            if hasattr(DikeSection,'Kruinhoogte'):
                if DikeSection.Kruinhoogte != np.max(DikeSection.InitialGeometry.z):
                    #In case the crest is unequal to the Kruinhoogte, that value should be given as input as well
                    self.measures[-1]['Geometry'], area_extra,area_excavated, dhouse = DetermineNewGeometry(j,self.parameters['Direction'],self.parameters['max_outward'],copy.deepcopy(DikeSection.InitialGeometry),
                                                                                                            self.geometry_plot, **{'plot_dir': plot_dir, 'slope_in': slope_in, 'crest_extra':DikeSection.Kruinhoogte})
                else:
                    self.measures[-1]['Geometry'], area_extra,area_excavated, dhouse = DetermineNewGeometry(j,self.parameters['Direction'],self.parameters['max_outward'],copy.deepcopy(DikeSection.InitialGeometry),
                                                                                                            self.geometry_plot, **{'plot_dir': plot_dir, 'slope_in': slope_in})
            else:
                self.measures[-1]['Geometry'], area_extra,area_excavated, dhouse = DetermineNewGeometry(j,self.parameters['Direction'],self.parameters['max_outward'],copy.deepcopy(DikeSection.InitialGeometry),
                                                                                                        self.geometry_plot, **{'plot_dir': plot_dir, 'slope_in': slope_in})

            self.measures[-1]['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, self.unit_costs, dcrest = j[0], dberm_in =int(dhouse), housing = DikeSection.houses, area_extra= area_extra, area_excavated = area_excavated,direction = self.parameters['Direction'],section=DikeSection.name)
            self.measures[-1]['Reliability'] = SectionReliability()
            self.measures[-1]['Reliability'].Mechanisms = {}

            for i in mechanisms:
                calc_type = DikeSection.MechanismData[i][1]
                self.measures[-1]['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, measure_year=self.parameters['year'])
                for ij, reliability_input in self.measures[-1]['Reliability'].Mechanisms[i].Reliability.items():
                    #for all time steps considered.
                    #first copy the data
                    reliability_input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                    #Adapt inputs for reliability calculation, but only after year of implementation.
                    if float(ij) >= self.parameters['year']:
                        reliability_input.input = implement_berm_widening(input=reliability_input.input,measure_input = self.measures[-1],measure_parameters = self.parameters, mechanism=i,computation_type = calc_type)
                    #put them back in the object
                    self.measures[-1]['Reliability'].Mechanisms[i].Reliability[ij].Input = reliability_input
                self.measures[-1]['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
            self.measures[-1]['Reliability'].calcSectionReliability()

class DiaphragmWall(Measure):
    # type == 'Diaphragm Wall':
    def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        #StabilityInner and Piping reduced to 0, height is ok for overflow until 2125 (free of charge, also if there is a large height deficit).
        # It is assumed that the diaphragm wall is extendable after that.
        #Only 1 parameterized version with a lifetime of 100 years
        self.measures = {}
        self.measures['DiaphragmWall'] = 'yes'
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, self.unit_costs)
        self.measures['Reliability'] = SectionReliability()
        self.measures['Reliability'].Mechanisms = {}
        for i in mechanisms:
            calc_type = DikeSection.MechanismData[i][1]
            self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type)
            for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                if float(ij) >= self.parameters['year']:
                    if i == 'Overflow':
                        Pt = TrajectInfo['Pmax']*TrajectInfo['omegaOverflow']
                        if DikeSection.Reliability.Mechanisms[i].Reliability[ij].type == 'Simple':
                            if hasattr(DikeSection,'HBNRise_factor'):
                                hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange = DikeSection.HBNRise_factor * DikeSection.YearlyWLRise, mechanism='Overflow')
                            else:
                                hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange=None, mechanism='Overflow')
                        else:
                            hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange=None, type='HRING', mechanism='Overflow')

                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['h_crest'] = \
                            np.max([hc, self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input[
                                'h_crest']])  #should not become weaker!
                    elif i == 'StabilityInner' or i == 'Piping':
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Elimination'] = 'yes'
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_elim'] = self.parameters['P_solution']
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_with_elim'] = self.parameters['Pf_solution']
            self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
        self.measures['Reliability'].calcSectionReliability()

class StabilityScreen(Measure):
    # type == 'Stability Screen':
    def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False, SFincrease = 0.2):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        self.measures = {}
        self.measures['Stability Screen'] = 'yes'
        if 'd_cover' in DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input:
            self.parameters['Depth'] = np.max([DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input['d_cover'] + 1., 8.])
        else:
            #TODO remove shaky assumption on depth
            self.parameters['Depth'] = 6.
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, self.unit_costs)
        self.measures['Reliability'] = SectionReliability()
        self.measures['Reliability'].Mechanisms = {}
        for i in mechanisms:
            calc_type = DikeSection.MechanismData[i][1]
            self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type)
            for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                if i == 'Overflow' or i == 'Piping': #Copy results
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij] = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij])
                    pass #no influence
                elif i == 'StabilityInner':
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                    if int(ij)>=self.parameters['year']:
                        if 'SF_2025' in self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input:
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] += SFincrease
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] += SFincrease
                        elif 'beta_2025' in self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input:
                            #convert to SF and back:
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2025'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2025'], type = 'beta'), SFincrease), type = 'SF')
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2075'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2075'], type = 'beta'), SFincrease), type = 'SF')
                        else:
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF'] = np.add(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF'], SFincrease)
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['BETA'] = beta_SF_StabilityInner(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF'], type = 'SF')

            self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
        self.measures['Reliability'].calcSectionReliability()


class VerticalGeotextile(Measure):
    def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()

        # No influence on overflow and stability
        # Only 1 parameterized version with a lifetime of 50 years
        self.measures = {}
        self.measures['VZG'] = 'yes'
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, self.unit_costs)
        self.measures['Reliability'] = SectionReliability()
        self.measures['Reliability'].Mechanisms = {}

        for i in mechanisms:
            calc_type = DikeSection.MechanismData[i][1]
            self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type)
            for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input = copy.deepcopy(
                    DikeSection.Reliability.Mechanisms[i].Reliability[ij].Input)
                if i == 'Overflow' or i == 'StabilityInner' or (
                        i == 'Piping' and int(ij) < self.parameters['year']):  # Copy results
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij] = copy.deepcopy(
                        DikeSection.Reliability.Mechanisms[i].Reliability[ij])
                elif i == 'Piping' and int(ij) >= self.parameters['year']:
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Elimination'] = 'yes'
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_elim'] = self.parameters[
                        'P_solution']
                    self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Pf_with_elim'] = \
                    np.min([self.parameters['Pf_solution'], 1.e-16])
            self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load, mechanism=i,
                                                                          trajectinfo=TrajectInfo)
        self.measures['Reliability'].calcSectionReliability()


class CustomMeasure(Measure):
    def set_input(self,section):
        try:
            try:
                data = pd.read_csv(self.input_directory.joinpath('Measures', self.parameters['File']))
            except:
                data = pd.read_csv(self.parameters['File'])
            reliability_headers = []
            for i, element in enumerate(list(data.columns)):
                #find and split headers
                if 'beta' in element:
                    reliability_headers.append(element.split('_'))
                    if 'start_id' not in locals():
                        start_id =i
            #make 2 dataframes: 1 with base data and 1 with reliability data
            base_data = data.iloc[:,0:start_id]
            reliability_data = data.iloc[:,start_id:]
            reliability_data.columns = pd.MultiIndex.from_arrays([np.array(reliability_headers)[:,1],np.array(reliability_headers)[:,2].astype(np.int32)],names=['mechanism','year'])
            #TODO reindex the reliability data such that the mechanism is the index and year the column. Now it is a multiindex, hwich works as well but is not as nice.
        except:
            raise Exception(self.parameters['File'] + ' not found.')
        # self.base_data = base_data
        self.reliability_data = reliability_data
        self.measures = {}
        self.parameters['year'] = np.int32(base_data['year'] - self.t_0)

        #TODO check these values:
        #for testing:
        # print('test values in Custom Measure')
        # base_data['kruinhoogte']=6.
        # base_data['extra kwelweg'] = 10.
        annual_dhc = section.Reliability.Mechanisms['Overflow'].Reliability['0'].Input.input['dhc(t)']
        if base_data['kruinhoogte_2075'].values > 0:
            self.parameters['h_crest_new'] = base_data['kruinhoogte_2075'].values + 50 * annual_dhc
        else:
            self.parameters['h_crest_new'] = None
        # print('Warning: kruinhoogte of custom measure should be adapted')
        #TODO modify kruinhoogte_2075 to 2025 using change of crest in time.
        self.parameters['L_added'] = base_data['verlenging kwelweg']
        self.measures['Cost'] = base_data['cost'].values[0]
        self.measures['Reliability'] = SectionReliability()
        self.measures['Reliability'].Mechanisms = {}

    def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        mechanisms = list(DikeSection.Reliability.Mechanisms.keys())

        #first read and set the data:
        self.set_input(DikeSection)

        #loop over mechanisms to modify the reliability
        for i in mechanisms:


            self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, computation_type=False)
            for ij in self.measures['Reliability'].Mechanisms[i].Reliability.keys():
                self.measures['Reliability'].Mechanisms[i].Reliability[ij] = copy.deepcopy(DikeSection.Reliability.Mechanisms[i].Reliability[ij])

                #only adapt after year of implementation:
                if np.int(ij) >= self.parameters['year']:
                    #remove other input:
                    if i == 'Overflow':
                        if self.parameters['h_crest_new'] != None:
                            #type: simple
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['h_crest'] = self.parameters['h_crest_new']

                        #change crest
                    elif i == 'Piping':
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['Lvoor'] += self.parameters['L_added'].values
                        #change Lvoor
                    else:
                        #Direct input: remove existing inputs and replace with beta
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].type = 'DirectInput'
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input = {}
                        self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta'] = {}
                        for input in self.reliability_data[i]:
                            #only read non-nan values:
                            if not np.isnan(self.reliability_data[i, input].values[0]):
                                self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta'][input-self.t_0] = self.reliability_data[i, input].values[0]
            self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
        self.measures['Reliability'].calcSectionReliability()


def implement_berm_widening(input, measure_input, measure_parameters, mechanism,computation_type, SFincrease = 0.2):
    # this function implements a berm widening based on the relevant inputs
    if mechanism == 'Overflow':
        input['h_crest'] = input['h_crest'] + measure_input['dcrest']
    elif mechanism == 'StabilityInner':
        #For stability factors
        if 'SF_2025' in input:
            #For now, inward and outward are the same!
            if (measure_parameters['Direction'] == 'inward') or (measure_parameters['Direction'] == 'outward'):
                input['SF_2025'] = input['SF_2025'] + (measure_input['dberm'] * input['dSF/dberm'])
                input['SF_2075'] = input['SF_2075'] + (measure_input['dberm'] * input['dSF/dberm'])
            if measure_parameters['StabilityScreen'] == 'yes':
                input['SF_2025'] += SFincrease
                input['SF_2075'] += SFincrease
        #For betas as input
        elif 'beta_2025' in input:
            input['beta_2025'] = input['beta_2025'] + (measure_input['dberm'] * input['dbeta/dberm'])
            input['beta_2075'] = input['beta_2075'] + (measure_input['dberm'] * input['dbeta/dberm'])
            if measure_parameters['StabilityScreen'] == 'yes':
                # convert to SF and back:
                input['beta_2025'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(input['beta_2025'], type = 'beta'),SFincrease), type='SF')
                input['beta_2075'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(input['beta_2075'], type = 'beta'),SFincrease), type='SF')
        elif 'BETA' in input:
            #TODO make sure input is grabbed properly. Should be read from input sheet
            input['SF'] = input['SF'] + (.02 *measure_input['dberm'])
            input['BETA'] = input['BETA'] + (.13 *measure_input['dberm'])
            if measure_parameters['StabilityScreen'] == 'yes':
                # convert to SF and back:
                input['SF'] = beta_SF_StabilityInner(np.add(input['SF'],SFincrease), type='SF')
                input['BETA'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(input['BETA'], type = 'beta'),SFincrease), type='SF')
        #For fragility curve as input
        elif computation_type== 'FragilityCurve':
            raise Exception('Not implemented')
            #TODO Here we can develop code to add berms to sections with a fragility curve.
        else:
            raise Exception('Unknown input data for stability when widening the berm')

    elif mechanism == 'Piping':
        input['Lvoor'] = input['Lvoor'] + measure_input['dberm']
        # input['Lachter'] = np.max([0., input['Lachter'] - measure_input['dberm']])
        input['Lachter'] = (input['Lachter'] - measure_input['dberm']).clip(0)
    return input


def addBerm(initial, geometry, new_geometry, bermheight, dberm):
    i = int(initial[initial.type == 'innertoe'].index.values)
    j = int(initial[initial.type == 'innercrest'].index.values)
    if (initial.type == 'extra').any():
        new_geometry[0][0] = new_geometry[0][0] -100

    slope_inner = (geometry[j][1] - geometry[i][1]) / (geometry[j][0] - geometry[i][0])
    extra = np.empty((1, 2))
    extra[0, 0] = new_geometry[i][0] + (1 / slope_inner) * bermheight
    extra[0, 1] = new_geometry[i][1] + bermheight
    new_geometry = np.append(new_geometry, np.array(extra), axis=0)
    extra2 = np.empty((1, 2))
    extra2[0, 0] = new_geometry[i][0] + (1 / slope_inner) * bermheight + dberm
    extra2[0, 1] = new_geometry[i][1] + bermheight
    new_geometry = np.append(new_geometry, np.array(extra2), axis=0)
    new_geometry = new_geometry[new_geometry[:, 0].argsort()]
    if (initial.type == 'extra').any():
        new_geometry[0][0] = new_geometry[0][0] +100
    return new_geometry

def addExtra(initial, new_geometry):
    i = int(initial[initial.type == 'innertoe'].index.values)
    k = int(initial[initial.type == 'extra'].index.values)
    new_geometry[0, 0] = initial.x[i]
    new_geometry[0, 1] = initial.z[i]
    extra3 = np.empty((1, 2))
    extra3[0, 0] = initial.x[k]
    extra3[0, 1] = initial.z[k]
    new_geometry = np.append(np.array(extra3), new_geometry, axis=0)
    return new_geometry


def calculateArea(geometry):
    polypoints = []
    for label, points in geometry.iterrows():
        polypoints.append((points.x, points.z))
    polygonXZ = Polygon(polypoints)
    areaPol = Polygon(polygonXZ).area
    return areaPol, polygonXZ

def ModifyGeometryInput(initial,bermheight):
    '''Checks geometry and corrects if necessary'''
    #TODO move this to the beginning for the input.
    #modify the old structure
    if not 'BUK' in initial.index:
        initial = initial.replace({'innertoe':'BIT','innerberm1':'EBL','innerberm2':'BBL','innercrest':'BIK','outercrest':'BUK','outertoe':'BUT'}).reset_index().set_index('type')


    if initial.loc['BUK'].x != 0.0:
        #if BUK is not at x = 0 , modify entire profile
        initial['x'] = np.subtract(initial['x'],initial.loc['BUK'].x)

    if initial.loc['BUK'].x > initial.loc['BIK'].x:
        #BIK must have larger x than BUK, so likely the profile is mirrored, mirror it back:
        initial['x'] = np.multiply(initial['x'], -1.)
    #if EBL and BBL not there, generate them.
    if not 'EBL' in initial.index:
        inner_slope = np.abs(initial.loc['BIT'].z -initial.loc['BIK'].z)/np.abs(initial.loc['BIT'].x -initial.loc['BIK'].x)
        initial.loc['EBL','x'] = initial.loc['BIT'].x - (bermheight/inner_slope)
        initial.loc['BBL','x'] = initial.loc['BIT'].x - (bermheight/inner_slope)
        initial.loc['BBL','z'] = initial.loc['BIT'].z + bermheight
        initial.loc['EBL','z'] = initial.loc['BIT'].z + bermheight

    return initial

#This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def DetermineNewGeometry(geometry_change, direction, maxbermout, initial, geometry_plot: bool, plot_dir = None, bermheight = 2, slope_in = False, crest_extra = False):
    '''initial should be a DataFrame with index values BUT, BUK, BIK, BBL, EBL and BIT.
    If this is not the case and it is input of the old type, first it is transformed to obey that.
    crest_extra is an additional argument in case the crest height for overflow is higher than the BUK and BIT.
    In such cases the crest heightening is the given increment + the difference between crest_extra and the BUK/BIT, such that after reinforcement the height is crest_extra + increment.
    It has to be ensured that the BUK has x = 0, and that x increases inward'''
    initial = ModifyGeometryInput(initial,bermheight)
    # maxBermOut=20
    # if len(initial) == 6:
    #     noberm = False
    # elif len(initial) == 4:
    #     noberm=True
    # else:
    #     raise Exception ('input length dike is not 4 or 6')

    # if z innertoe != z outertoe add a point to ensure correct shapely operations
    initial.loc['EXT','x'] = initial.loc['BIK'].x
    initial.loc['EXT','z'] = np.min(initial.z)

    if initial.loc['BIT'].z > initial.loc['BUT'].z:
        initial.loc['BIT_0','x'] = initial.loc['BIT'].x
        initial.loc['BIT_0','z'] = initial.loc['BIT'].z
        initial=initial.reindex(['BUT', 'BUK', 'BIK','BBL','EBL','BIT','BIT_0','EXT'])
    elif initial.loc['BIT'].z < initial.loc['BUT'].z:
        initial.loc['BUT_0','x'] = initial.loc['BUT'].x
        initial.loc['BUT_0','z'] = initial.loc['BUT'].z
        initial=initial.reindex(['BUT', 'BUT_0','BUK', 'BIK','BBL','EBL','BIT','EXT'])



    # Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm = geometry_change[1]
    if crest_extra:
        if (crest_extra > initial['z'].max()) & (dcrest >0.):
            #if overflow crest is higher than profile, in case of reinforcement ensure that everything is heightened to that level + increment:
            pass
        elif (crest_extra < initial['z'].max()):
            #case where cross section for overflow has a lower spot, but majority of section is higher.
            #in that case the crest height is modified to the level of the overflow computation which is a conservative estimate.
            initial.loc['BIK','z'] = crest_extra
            initial.loc['BUK','z'] = crest_extra
        cur_crest = crest_extra

    else:
        cur_crest = initial['z'].max()
    new_crest = cur_crest + dcrest

    #crest heightening
    if dcrest > 0:
        #determine widening at toes.
        slope_out = np.abs(initial.loc['BUK'].x - initial.loc['BUT'].x)/np.abs(initial.loc['BUK'].z - initial.loc['BUT'].z)
        BUT_dx = out = slope_out * dcrest

        #TODO discuss with WSRL: if crest is heightened, should slope be determined based on BIK and BIT or BIK and BBL?
        #Now it has been implemented that the slope is based on BIK and BBL
        slope_in = np.abs(initial.loc['BBL'].x - initial.loc['BIK'].x)/np.abs(initial.loc['BBL'].z - initial.loc['BIK'].z)
        BIT_dx = slope_in * dcrest
    else:
        BUT_dx = 0.
        BIT_dx = 0.
    # z_innertoe = (initial.z[int(initial[initial.type == 'innertoe'].index.values)])

    if direction == 'outward':
        warnings.warn('Outward reinforcement is not updated!')
        # nieuwe opzet:
        # if outward:
        #    verplaats buitenkruin en buitenteen
        #   ->tussen geometrie 1
        #  afgraven
        # ->tussen geometrie 2

        # berm er aan plakken. Ook bij alleen binnenwaarts

        # volumes berekenen (totaal extra, en totaal "verplaatst in profiel")

        # optional extension: optimize amount of outward/inward reinforcement
        new_geometry = copy.deepcopy(initial)

        if dberm <= maxbermout:

            for count, i in new_geometry.iterrows():
                # Run over points
                if initial.type[i] == 'extra':
                    new_geometry[i][0] = geometry[i][0]
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innertoe':
                    new_geometry[i][0] = geometry[i][0] + dberm + dout - din
                    new_geometry[i][1] = geometry[i][1]
                    dhouse = max(0,-(dberm + dout - din))
                elif initial.type[i] == 'innerberm1':
                    new_geometry[i][0] = geometry[i][0] + dberm + dout - din
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm2':
                    new_geometry[i][0] = geometry[i][0] + dberm + dout - din
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innercrest':
                    new_geometry[i][0] = geometry[i][0] + dberm + dout
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outercrest':
                    new_geometry[i][0] = geometry[i][0] + dberm + dout
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outertoe':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'extra2':
                    new_geometry[i][0] = geometry[i][0] + dberm
                    new_geometry[i][1] = geometry[i][1]
            if (initial.type == 'extra').any():
                if dberm > 0 or dcrest > 0:
                    new_geometry = addExtra(initial, new_geometry)

        else:
            berm_in = dberm - maxbermout
            for i in range(len(new_geometry)):
                # Run over points
                if initial.type[i] == 'extra':
                    new_geometry[i][0] = geometry[i][0]
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innertoe':
                    new_geometry[i][0] = geometry[i][0] - berm_in + dout - din
                    new_geometry[i][1] = geometry[i][1]
                    dhouse = max(0,-(- berm_in + dout - din))
                elif initial.type[i] == 'innerberm1':
                    new_geometry[i][0] = geometry[i][0] - berm_in + dout - din
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innerberm2':
                    new_geometry[i][0] = geometry[i][0] + maxbermout + dout - din
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'innercrest':
                    new_geometry[i][0] = geometry[i][0] + maxbermout + dout
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outercrest':
                    new_geometry[i][0] = geometry[i][0] + maxbermout + dout
                    new_geometry[i][1] = geometry[i][1] + dcrest
                elif initial.type[i] == 'outertoe':
                    new_geometry[i][0] = geometry[i][0] + maxbermout
                    new_geometry[i][1] = geometry[i][1]
                elif initial.type[i] == 'extra2':
                    new_geometry[i][0] = geometry[i][0] + maxbermout
                    new_geometry[i][1] = geometry[i][1]
            # if noberm:  # len(initial) == 4:
            #     if dberm > 0:
            #         new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)
            # if (initial.type == 'extra').any():
            #     if dberm > 0 or dcrest > 0:
            #         new_geometry = addExtra(initial, new_geometry)

    if direction == 'inward':
        # all changes inward.
        new_geometry = copy.deepcopy(initial)
        for ind, data in new_geometry.iterrows():
            # Run over points .
            if ind in ['EXT','BUT', 'BUT_0', 'BIT_0']: #Points that are not modified
                xz = data.values
            if ind == 'BIT':
                xz = [data.x + dberm + BUT_dx + BIT_dx, data.z]
                dhouse = max(0,dberm + BUT_dx + BIT_dx)
            elif ind == 'EBL':
                xz = [data.x + dberm + BUT_dx + BIT_dx, data.z]
            elif ind == 'BBL':
                xz = [data.x + BUT_dx + BIT_dx, data.z]
            elif ind == 'BIK':
                xz = [data.x + BUT_dx, data.z + dcrest]
            elif ind == 'BUK':
                xz = [data.x + BUT_dx, data.z + dcrest]
            new_geometry.loc[ind] = pd.Series(xz,index=['x','z'])

    # calculate the area difference
    area_old, polygon_old = calculateArea(initial)
    area_new, polygon_new = calculateArea(new_geometry)
    #
    # plt.plot(initial.x,initial.z, 'ko')
    # plt.plot(*polygon_old.exterior.xy, 'g')
    # plt.plot(*polygon_new.exterior.xy, 'r--')
    # plt.savefig('testgeom.png')
    # plt.close()
    if (polygon_old.intersects(polygon_new)):  # True
        try:
            poly_intsects = polygon_old.intersection(polygon_new)
        except:
            plt.plot(initial.x,initial.z, 'ko')
            plt.plot(*polygon_old.exterior.xy, 'g')
            plt.plot(*polygon_new.exterior.xy, 'r--')
            plt.savefig('testgeom.png')
            plt.close()
        area_intersect = (polygon_old.intersection(polygon_new).area)  # 1.0
        area_excavate = area_old - area_intersect
        area_extra = area_new - area_intersect

        #difference new-old = extra
        poly_diff = polygon_new.difference(polygon_old)
        area_diff = poly_diff.area  #zou zelfde moeten zijn als area_extra
        # difference new-old = excavate
        poly_diff2 = polygon_old.difference(polygon_new)
        area_diff2 = poly_diff2.area  #zou zelfde moeten zijn als area_excavate

        #controle
        test1 = area_diff - area_extra
        test2 = area_diff2 - area_excavate
        if test1>1 or test2 >1:
            raise Exception ('area calculation failed')

        if geometry_plot:
            if not plot_dir.joinpath('Geometry').is_dir():
                # plot_dir.joinpath.mkdir(parents=True, exist_ok=True)
                plot_dir.joinpath('Geometry').mkdir(parents=True, exist_ok=True)
            plt.plot(geometry[:, 0], geometry[:, 1], 'k')
            plt.plot(new_geometry[:, 0], new_geometry[:, 1], '--r')
            if poly_diff.area > 0:
                if hasattr(poly_diff, 'geoms'):
                    for i in range(len(poly_diff.geoms)):
                        x1, y1 = poly_diff[i].exterior.xy
                        plt.fill(x1, y1, 'r--', alpha=.1)
                else:
                    x1, y1 = poly_diff.exterior.xy
                    plt.fill(x1, y1, 'r--', alpha=.1)
            if poly_diff2.area > 0:
                if hasattr(poly_diff2, 'geoms'):
                    for i in range(len(poly_diff2.geoms)):
                        x1, y1 = poly_diff2[i].exterior.xy
                        plt.fill(x1, y1, 'b--', alpha=.8)
                else:
                    x1, y1 = poly_diff2.exterior.xy
                    plt.fill(x1, y1, 'b--', alpha=.8)            #
            # if hasattr(poly_intsects, 'geoms'):
            #     for i in range(len(poly_intsects.geoms)):
            #         x1, y1 = poly_intsects[i].exterior.xy
            #         plt.fill(x1, y1, 'g--', alpha=.1)
            # else:
            #     x1, y1 = poly_intsects.exterior.xy
            #     plt.fill(x1, y1, 'g--', alpha=.1)
            # plt.show()

            plt.text(np.mean(new_geometry[:, 0]), np.max(new_geometry[:, 1]),'Area extra = {:.4} $m^2$, area excavated = {:.4} $m^2$'.format(str(area_extra),str(area_excavate)))

            plt.savefig(plot_dir.joinpath('Geometry_' + str(dberm) + '_' + str(dcrest)+ direction + '.png'))
            plt.close()

    area_difference = np.max([0.,area_extra + 0.5 * area_excavate])
    #old:
    # return new_geometry, area_difference
    return new_geometry, area_extra, area_excavate, dhouse


#Script to determine the costs of a reinforcement:
def DetermineCosts(parameters, type, length, unit_costs:dict, dcrest = 0., dberm_in = 0., housing = False, area_extra = False, area_excavated = False, direction = False, section = ''):
    if (type == 'Soil reinforcement') and (direction == 'outward') and (dberm_in >0.):
        #as we only use unit costs for outward reinforcement, and these are typically lower, the computation might be incorrect (too low).
        print('Warning: encountered outward reinforcement with inward berm. Cost computation might be inaccurate')
    if type == 'Soil reinforcement':
     if direction == 'inward':
         C = unit_costs['Inward added volume'] * area_extra * length + unit_costs['Inward starting costs'] *length
     elif direction == 'outward':
         volume_excavated = area_excavated * length
         volume_extra = area_extra *length
         reusable_volume = unit_costs['Outward reuse factor'] * volume_excavated
         #excavate and remove part of existing profile:
         C = unit_costs['Outward removed volume'] * (volume_excavated-reusable_volume)

         #apply reusable volume
         C += unit_costs['Outward reused volume'] * reusable_volume
         remaining_volume = volume_extra - reusable_volume

         #add additional soil:
         C += unit_costs['Outward added volume'] * remaining_volume

         #compensate:
         C += unit_costs['Outward removed volume'] * unit_costs['Outward compensation factor'] * volume_extra


     else:
         raise Exception('invalid direction')

     #add costs for housing
     if isinstance(housing, pd.DataFrame) and dberm_in > 0.:
         if dberm_in > housing.size:
            warnings.warn('Inwards reinforcement distance exceeds data for housing database at section {}'.format(section))
            # raise Exception('inwards distance exceeds housing database')
            C += parameters['C_house'] * housing.loc[housing.size]['cumulative']
         else:
            C += parameters['C_house'] * housing.loc[float(dberm_in)]['cumulative']

    #add costs for stability screen
     if parameters['StabilityScreen'] == 'yes':
         C += unit_costs['Sheetpile'] * parameters['Depth'] * length

     if dcrest >0.:
         C += unit_costs['Road renewal'] * length

     #x = map(int, self.parameters['house_removal'].split(';'))
    elif type == 'Vertical Geotextile':
     C = unit_costs['Vertical Geotextile'] * length
    elif type == 'Diaphragm Wall':
     C = unit_costs['Diaphragm wall'] * length
    elif type == 'Stability Screen':
     C = unit_costs['Sheetpile'] * parameters['Depth'] * length
    else:
     print('Unknown type')
    return C

#Script to determine the required crest height for a certain year
def ProbabilisticDesign(design_variable, strength_input, Pt, horizon = 50, loadchange = 0, mechanism='Overflow',type = 'SAFE'):
 if mechanism == 'Overflow':
     if type == 'SAFE':
         #determine the crest required for the target
         h_crest, beta = OverflowSimple(strength_input['h_crest'], strength_input['q_crest'], strength_input['h_c'], strength_input['q_c'], strength_input['beta'], mode='design', Pt=Pt, design_variable=design_variable)
         #add temporal changes due to settlement and climate change
         h_crest = h_crest + horizon * (strength_input['dhc(t)'] + loadchange)
         return h_crest
     elif type == 'HRING':
         h_crest, beta = OverflowHRING(strength_input, horizon, mode='design', Pt = Pt)
         return h_crest
     else:
         raise Exception('Unknown calculation type for {}'.format(mechanism))