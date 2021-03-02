import numpy as np
import copy
import matplotlib.pyplot as plt
from Mechanisms import OverflowSimple
from ReliabilityCalculation import MechanismReliabilityCollection
from SectionReliability import SectionReliability
from shapely.geometry import Polygon
import pandas as pd
import config

"""Important: a measure is a single type of reinforcement, so for instance a stability screen. A solution can be a COMBINATION of measures (e.g. a stability screen with a berm)"""
class Measure:
    """Possible change: create subclasses for different measures to make the below code more neat. Can be done jointly with adding outward reinforcement"""
    #class to store measures and their reliability. A Measure is a specific Solution (with parameters)
    def __init__(self,inputs):
        self.parameters = {}
        for i in range(0,len(inputs)):
            if ~(inputs[i] is np.nan or inputs[i] != inputs[i]):
                self.parameters[inputs.index[i]] = inputs[i]

    def evaluateMeasure(self,DikeSection,TrajectInfo,preserve_slope = False):
        from HelperFunctions import createDir

        #To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters['Type']
        mechanisms = DikeSection.Reliability.Mechanisms.keys()
        SFincrease = 0.2        #for stability screen
        if config.geometry_plot:
            plt.figure(1000)
            createDir(config.directory.joinpath('figures',DikeSection.name,'Geometry'))

        #different types of measures:
        if type == 'Soil reinforcement':
            crest_step = 0.5
            berm_step = 10
            crestrange = np.linspace(self.parameters['dcrest_min'], self.parameters['dcrest_max'], np.int(1 + (self.parameters['dcrest_max']-self.parameters['dcrest_min']) / crest_step))
            if self.parameters['Direction'] == 'outward':
                bermrange = np.linspace(0., self.parameters['max_outward'], np.int(1+(self.parameters['max_outward']/berm_step)))
            elif self.parameters['Direction'] == 'inward':
                bermrange = np.linspace(0., self.parameters['max_inward'], np.int(1+(self.parameters['max_inward']/berm_step)))
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
                self.measures.append({})
                self.measures[-1]['dcrest'] =j[0]
                self.measures[-1]['dberm'] = j[1]
                self.measures[-1]['Geometry'], area_difference = DetermineNewGeometry(j,self.parameters['Direction'],DikeSection.InitialGeometry, plot_dir = config.directory.joinpath('figures', DikeSection.name, 'Geometry'), slope_in = slope_in)
                self.measures[-1]['Cost'] = DetermineCosts(self.parameters, type, DikeSection.Length, reinf_pars = j, housing = DikeSection.houses, area_difference= area_difference)
                self.measures[-1]['Reliability'] = SectionReliability()
                self.measures[-1]['Reliability'].Mechanisms = {}

                for i in mechanisms:
                    calc_type = DikeSection.MechanismData[i][1]
                    self.measures[-1]['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type, measure_year=self.parameters['year'])
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
                self.measures[-1]['Reliability'].calcSectionReliability()
                #TODO add interpolation option here (and for the other types)
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
                self.measures['Reliability'].Mechanisms[i] = MechanismReliabilityCollection(i, calc_type)
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
            self.measures['Reliability'].calcSectionReliability()
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
        elif type == 'Stability Screen':
            self.measures = {}
            self.measures['Stability Screen'] = 'yes'
            SFincrease = 0.2
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
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2025'] += SFincrease
                            self.measures['Reliability'].Mechanisms[i].Reliability[ij].Input.input['SF_2075'] += SFincrease
                self.measures['Reliability'].Mechanisms[i].generateLCRProfile(DikeSection.Reliability.Load,mechanism=i,trajectinfo=TrajectInfo)
            self.measures['Reliability'].calcSectionReliability()
        elif type == 'Custom':
            try:
                data = pd.read_csv(config.path.joinpath('Measures',self.parameters['File']))
                print('Here the logic for reading the data for the vka should be inserted')
                #interpret data
            except:
                raise Exception (self.parameters['File'] + ' not found.')

#This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def DetermineNewGeometry(geometry_change, direction, initial,plot_dir = None, bermheight = 2, slope_in = False):
    if len(initial) == 6:
        bermheight = initial.iloc[2]['z']
    elif len(initial) == 4:
        pass
    #Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm  = geometry_change[1]
    geometry = initial.values
    cur_crest = np.max(geometry[:,1])
    new_crest = cur_crest+dcrest
    if config.geometry_plot:
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

        if config.geometry_plot:
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
    if config.geometry_plot:
        if hasattr(poly_diff, 'geoms'):
            for i in range(len(poly_diff.geoms)):
                x1, y1 = poly_diff[i].exterior.xy
                plt.fill(x1, y1, 'r--')

        else:
            x1, y1 = poly_diff.exterior.xy
            plt.fill(x1, y1, 'r--')

        plt.text(np.mean(new_geometry[:, 0]), np.max(new_geometry[:, 1]), 'Area difference = ' + '{:.4}'.format(str(area_difference)) + ' $m^2$')

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