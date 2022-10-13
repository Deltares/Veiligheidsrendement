import numpy as np
import copy
import matplotlib.pyplot as plt
from FloodDefenceSystem.Mechanisms import OverflowSimple
from FloodDefenceSystem.ReliabilityCalculation import MechanismReliabilityCollection, beta_SF_StabilityInner
from FloodDefenceSystem.SectionReliability import SectionReliability
from shapely.geometry import Polygon
import pandas as pd
import config

"""Important: a measure is a single type of reinforcement, so for instance a stability screen. A solution can be a COMBINATION of measures (e.g. a stability screen with a berm)"""
class Measure():
    """Possible change: create subclasses for different measures to make the below code more neat. Can be done jointly with adding outward reinforcement"""
    #class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    def __init__(self, inputs):
        self.parameters = {}
        for i in range(0,len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]

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
    def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
    # def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        #To be added: year property to distinguish the same measure in year 2025 and 2045
        # Measure.__init__(self,inputs)
        # self. parameters = measure.parameters

        SFincrease = 0.2  # for stability screen

        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        crest_step = config.crest_step
        berm_step = config.berm_step
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
            self.parameters['Depth'] = np.max([DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability[
                                           '0'].Input.input['d_cover'] + 1., 8.])

        for j in measures:
            if self.parameters['Direction'] == 'outward':
                k = max(0, j[1]-self.parameters['max_inward']) #correction max_outward
            else:
                k = j[1]
            self.measures.append({})
            self.measures[-1]['dcrest'] =j[0]
            self.measures[-1]['dberm'] = j[1]
            self.measures[-1]['Geometry'], area_extra,area_excavated, dhouse = DetermineNewGeometry(j,self.parameters['Direction'],self.parameters['max_outward'],DikeSection.InitialGeometry, plot_dir = config.directory.joinpath('figures', DikeSection.name, 'Geometry'), slope_in = slope_in)
            self.measures[-1]['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, dcrest = j[0], dberm_in =int(dhouse), housing = DikeSection.houses, area_extra= area_extra, area_excavated = area_excavated,direction = self.parameters['Direction'])
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
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
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
                        if hasattr(DikeSection,'HBNRise_factor'):
                            hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange = DikeSection.HBNRise_factor * DikeSection.YearlyWLRise, mechanism='Overflow')
                        else:
                            hc = ProbabilisticDesign('h_crest', DikeSection.Reliability.Mechanisms['Overflow'].Reliability[ij].Input.input, Pt=Pt, horizon = self.parameters['year'] + 100, loadchange=None, mechanism='Overflow')
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
        self.parameters['Depth'] = np.max([DikeSection.Reliability.Mechanisms['StabilityInner'].Reliability['0'].Input.input['d_cover'] + 1., 8.])
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
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
                        else:
                            #convert to SF and back:
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2025'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2025'], type = 'beta'), SFincrease), type = 'SF')
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2075'] = beta_SF_StabilityInner(np.add(beta_SF_StabilityInner(self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta_2075'], type = 'beta'), SFincrease), type = 'SF')

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
        self.measures['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length)
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
            data = pd.read_csv(config.path.joinpath('Measures', self.parameters['File']))
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
        self.parameters['year'] = np.int32(base_data['year'] - config.t_0)

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
                                self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['beta'][input-config.t_0] = self.reliability_data[i, input].values[0]
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
        #For fragility curve as input
        elif computation_type== 'FragilityCurve':
            raise Exception('Not implemented')
            #TODO Here we can develop code to add berms to sections with a fragility curve.
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
    for i in range(len(geometry)):
        polypoints.append((geometry[i, 0], geometry[i, 1]))
    polygonXY = Polygon(polypoints)
    areaPol = Polygon(polygonXY).area
    return areaPol, polygonXY


#This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def DetermineNewGeometry(geometry_change, direction, maxbermout, initial,plot_dir = None, bermheight = 2, slope_in = False):
    # maxBermOut=20
    if len(initial) == 6:
        noberm = False
    elif len(initial) == 4:
        noberm=True
    else:
        raise Exception ('input length dike is not 4 or 6')

    # if outertoe < innertoe
    if initial.z[int(initial[initial.type == 'innertoe'].index.values)] > initial.z[int(initial[initial.type == 'outertoe'].index.values)]:
        extra_row = pd.DataFrame([[initial.x[int(initial[initial.type == 'innertoe'].index.values)],initial.z[int(initial[initial.type == 'outertoe'].index.values)], 'extra']],columns =initial.columns)
        initial = extra_row.append(initial).reset_index(drop=True)


    if initial.z[int(initial[initial.type == 'innertoe'].index.values)] < initial.z[int(initial[initial.type == 'outertoe'].index.values)]:
        extra_row2 = pd.DataFrame([[initial.x[int(initial[initial.type == 'outertoe'].index.values)],initial.z[int(initial[initial.type == 'innertoe'].index.values)], 'extra2']],columns =initial.columns)
        initial = initial.append(extra_row2).reset_index(drop=True)

    # Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm = geometry_change[1]
    geometry = initial.values[:,0:2]
    cur_crest = np.max(geometry[:, 1])
    new_crest = cur_crest + dcrest

    # if config.geometry_plot:
    #     plt.plot(geometry[:, 0], geometry[:, 1], 'k')
    if dcrest > 0:
        int_outercrest = int(initial[initial.type == 'outercrest'].index.values)
        int_outertoe = int(initial[initial.type == 'outertoe'].index.values)
        slope_out = ((initial.x[int_outercrest]) - (initial.x[int_outertoe])) / (
                    (initial.z[int_outercrest]) - (initial.z[int_outertoe]))
        dout = slope_out * dcrest
        int_innercrest = int(initial[initial.type == 'innercrest'].index.values)
        # int_innertoe = int(initial[initial.type == 'innertoe'].index.values)
        slope_inn = ((initial.x[int_innercrest]) - (initial.x[int_innercrest - 1])) / (
                    (initial.z[int_innercrest]) - (initial.z[int_innercrest - 1]))
        din = slope_inn * dcrest
    else:
        din = 0.
        dout = 0.
    z_innertoe = (initial.z[int(initial[initial.type == 'innertoe'].index.values)])

    if direction == 'outward':

        # nieuwe opzet:
        # if outward:
        #    verplaats buitenkruin en buitenteen
        #   ->tussen geometrie 1
        #  afgraven
        # ->tussen geometrie 2

        # berm er aan plakken. Ook bij alleen binnenwaarts

        # volumes berekenen (totaal extra, en totaal "verplaatst in profiel")

        # optional extension: optimize amount of outward/inward reinforcement
        new_geometry = copy.deepcopy(geometry)

        if dberm <= maxbermout:

            for i in range(len(new_geometry)):
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
            if noberm:  # len(initial) == 4:
                if dberm > 0:
                    new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)
            if (initial.type == 'extra').any():
                if dberm > 0 or dcrest > 0:
                    new_geometry = addExtra(initial, new_geometry)

    if direction == 'inward':
        new_geometry = copy.deepcopy(geometry)
        for i in range(len(new_geometry)):
            # Run over points .
            if initial.type[i] == 'extra':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innertoe':
                new_geometry[i][0] = geometry[i][0] - dberm + dout - din
                new_geometry[i][1] = geometry[i][1]
                dhouse = max(0,-(-dberm + dout - din))
            elif initial.type[i] == 'innerberm1':
                new_geometry[i][0] = geometry[i][0] - dberm + dout - din
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innerberm2':
                new_geometry[i][0] = geometry[i][0] + dout - din
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'innercrest':
                new_geometry[i][0] = geometry[i][0] + dout
                new_geometry[i][1] = geometry[i][1] + dcrest
            elif initial.type[i] == 'outercrest':
                new_geometry[i][0] = geometry[i][0] + dout
                new_geometry[i][1] = geometry[i][1] + dcrest
            elif initial.type[i] == 'outertoe':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]
            elif initial.type[i] == 'extra2':
                new_geometry[i][0] = geometry[i][0]
                new_geometry[i][1] = geometry[i][1]

        if noberm: #len(initial) == 4:   #precies hetzelfde als hierboven. def van maken.
            if dberm > 0:
                new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)

        if (initial.type == 'extra').any():
            if dberm > 0 or dcrest > 0:
                new_geometry = addExtra(initial, new_geometry)

    # calculate the area difference
    area_old, polygon_old = calculateArea(geometry)
    area_new, polygon_new = calculateArea(new_geometry)

    if (polygon_old.intersects(polygon_new)):  # True
        poly_intsects = polygon_old.intersection(polygon_new)
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

        if config.geometry_plot:
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
def DetermineCosts(parameters, type, length, dcrest = 0., dberm_in = 0., housing = False, area_extra = False, area_excavated = False, direction = False):
    if (type == 'Soil reinforcement') and (direction == 'outward') and (dberm_in >0.):
        #as we only use unit costs for outward reinforcement, and these are typically lower, the computation might be incorrect (too low).
        print('Warning: encountered outward reinforcement with inward berm. Cost computation might be inaccurate')
    if type == 'Soil reinforcement':
     if direction == 'inward':
         C = config.unit_cost['Inward added volume'] * area_extra * length + config.unit_cost['Inward starting costs'] *length
     elif direction == 'outward':
         volume_excavated = area_excavated * length
         volume_extra = area_extra *length
         reusable_volume = config.unit_cost['Outward reuse factor'] * volume_excavated
         #excavate and remove part of existing profile:
         C = config.unit_cost['Outward removed volume'] * (volume_excavated-reusable_volume)

         #apply reusable volume
         C += config.unit_cost['Outward reused volume'] * reusable_volume
         remaining_volume = volume_extra - reusable_volume

         #add additional soil:
         C += config.unit_cost['Outward added volume'] * remaining_volume

         #compensate:
         C += config.unit_cost['Outward removed volume'] * config.unit_cost['Outward compensation factor'] * volume_extra


     else:
         raise Exception('invalid direction')

     #add costs for housing
     if isinstance(housing, pd.DataFrame) and dberm_in > 0.:
         if dberm_in > housing.size:
             raise Exception('inwards distance exceeds housing database')

         C += parameters['C_house'] * housing.loc[float(dberm_in)]['cumulative']
    #add costs for stability screen
     if parameters['StabilityScreen'] == 'yes':
         C += config.unit_cost['Sheetpile'] * parameters['Depth'] * length

     if dcrest >0.:
         C += config.unit_cost['Road renewal'] * length

     #x = map(int, self.parameters['house_removal'].split(';'))
    elif type == 'Vertical Geotextile':
     C = config.unit_cost['Vertical Geotextile'] * length
    elif type == 'Diaphragm Wall':
     C = config.unit_cost['Diaphragm wall'] * length
    elif type == 'Stability Screen':
     C = config.unit_cost['Sheetpile'] * parameters['Depth'] * length
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