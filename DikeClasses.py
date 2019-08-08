import copy
import matplotlib.pyplot as plt
import Mechanisms
import numpy as np
import openturns as ot
import os
import pandas as pd
import ProbabilisticFunctions
from HydraRing_scripts import DesignTableOpenTurns
from ProbabilisticFunctions import TableDist, run_prob_calc, IterativeFC_calculation, TemporalProcess, addLoadCharVals, MarginalsforTimeDepReliability
from scipy.stats import norm
from scipy import interpolate


class DikeTraject:
    #This class contains general information on the dike traject and is used to store all data on the sections
    def __init__(self, name, traject=None):
        self.GeneralInfo = {}
        self.Sections = []

        # Basic traject info
        if traject == '16-4':
            self.GeneralInfo['FloodDamage'] = 23e9
            self.GeneralInfo['TrajectLength'] = 19480
            self.GeneralInfo['Pmax'] = 1. / 10000
            self.GeneralInfo['omegaPiping'] = 0.24; self.GeneralInfo['aPiping'] = 0.9; self.GeneralInfo['bPiping'] = 300
            self.GeneralInfo['omegaStabilityInner'] = 0.04; self.GeneralInfo['aStabilityInner'] = 0.033; self.GeneralInfo['bStabilityInner'] = 50
            self.GeneralInfo['omegaOverflow'] = 0.24;
        elif traject == '16-3':
            self.GeneralInfo['FloodDamage'] = 23e9
            self.GeneralInfo['TrajectLength'] = 19899
            self.GeneralInfo['Pmax'] = 1. / 10000
            self.GeneralInfo['omegaPiping'] = 0.24; self.GeneralInfo['aPiping'] = 0.9; self.GeneralInfo['bPiping'] = 300
            self.GeneralInfo['omegaStabilityInner'] = 0.04; self.GeneralInfo['aStabilityInner'] = 0.033; self.GeneralInfo['bStabilityInner'] = 50
            self.GeneralInfo['omegaOverflow'] = 0.24
            # NB: klopt a hier?????!!!!
        else:
            self.GeneralInfo['FloodDamage'] = 5e9
            self.GeneralInfo['Pmax'] = 1. / 10000
            self.GeneralInfo['omegaPiping'] = 0.24; self.GeneralInfo['aPiping'] = 0.9; self.GeneralInfo['bPiping'] = 300
            self.GeneralInfo['omegaStabilityInner'] = 0.04; self.GeneralInfo['aStabilityInner'] = 0.033; self.GeneralInfo['bStabilityInner'] = 50
            self.GeneralInfo['omegaOverflow'] = 0.24

    def ReadAllTrajectInput(self, path, directory, T, startyear,traject='16-4', mechanisms=['Overflow',
                                                                                           'StabilityInner',
                                                                                     'Piping']):
        #Make a case directory and inside a figures and results directory if it doesnt exist yet
        if not path.joinpath(directory).is_dir():
            path.joinpath(directory).mkdir(parents=True, exist_ok=True)
            path.joinpath(directory).joinpath('figures').mkdir(parents=True, exist_ok=True)
            path.joinpath(directory).joinpath('results', 'investment_steps').mkdir(parents=True, exist_ok=True)

        # Routine to read the input for all sections based on the default input format.
        files = [i for i in path.glob("*DV*") if i.is_file()]

        for i in range(len(files)):
            # Read the general information for each section:
            self.Sections.append(DikeSection(files[i].stem, traject))
            self.Sections[i].readGeneralInfo(path, 'General')

            # Read the data per mechanism, and first the load frequency line:
            self.Sections[i].Reliability.Load = LoadInput()
            self.Sections[i].Reliability.Load.set_fromDesignTable(path.joinpath('Toetspeil', self.Sections[i].LoadData))
            self.Sections[i].Reliability.Load.set_annual_change(type='SAFE', parameters=[self.Sections[i].YearlyWLRise, self.Sections[i].HBNRise_factor])

            # Then the input for all the mechanisms:
            self.Sections[i].Reliability.Mechanisms = {}
            for j in mechanisms:
                self.Sections[i].Reliability.Mechanisms[j] = MechanismReliabilityCollection(j, self.Sections[i].MechanismData[j][1], years=T)

                for k in self.Sections[i].Reliability.Mechanisms[j].Reliability.keys():
                    self.Sections[i].Reliability.Mechanisms[j].Reliability[k].Input.fill_mechanism(path.joinpath(
                        self.Sections[i].MechanismData[j][0]), calctype=self.Sections[i].MechanismData[j][1], mechanism=j)

            #Make in the figures directory a Initial and Measures direcotry if they don't exist yet
            if not path.joinpath(directory).joinpath('figures', self.Sections[i].name).is_dir():
                path.joinpath(directory).joinpath('figures', self.Sections[i].name, 'Initial').mkdir(parents=True, exist_ok=True)
                path.joinpath(directory).joinpath('figures', self.Sections[i].name, 'Measures').mkdir(parents=True, exist_ok=True)

        self.GeneralInfo['Mechanisms'] = mechanisms

        self.GeneralInfo['T'] = T
        self.GeneralInfo['StartYear'] = startyear
        #Traject length is lengt of all sections together:
        self.GeneralInfo['TrajectLength'] = 0
        for i in self.Sections:
            self.GeneralInfo['TrajectLength'] += i.Length

    def plotReliabilityofDikeTraject(self,
                                     PATH = None, fig_size = (6,4),
                                     draw_targetbeta='off', language='EN',
                                     flip='off', beta_or_prob = 'beta',
                                     outputcsv=False, first=True,
                                     last=True, alpha = 1, type = 'Assessment'):
        #a bunch of settings to make it look nice:
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        years = self.GeneralInfo['T']
        mechanisms = self.GeneralInfo['Mechanisms']
        startyear = self.GeneralInfo['StartYear']

        #read the assessments of all sections and write to one big dataframe:
        for i in range(0, len(self.Sections)):
            if i == 0:
                Assessment = self.Sections[i].Reliability.SectionReliability.reset_index()
                Assessment['Section'] = self.Sections[i].name
                Assessment['Length'] = self.Sections[i].Length
                Assessment.columns = Assessment.columns.astype(str)
                if 'mechanism' in Assessment.columns:
                    Assessment = Assessment.rename(columns={"mechanism":"index"})
            else:
                data_to_add = self.Sections[i].Reliability.SectionReliability.reset_index()
                data_to_add['Section'] = self.Sections[i].name
                data_to_add['Length'] = self.Sections[i].Length
                data_to_add.columns = data_to_add.columns.astype(str)
                if 'mechanism' in data_to_add.columns:
                    data_to_add = data_to_add.rename(columns={"mechanism": "index"})

                Assessment = Assessment.append(data_to_add, sort=False)

        Assessment = Assessment.reset_index(drop=True)
        if outputcsv:
            Assessment.to_csv(PATH.joinpath('AllBetas.csv'))

        #English or Dutch labels and titles
        if language == 'NL':
            label_xlabel = 'Dijkvakken'
            if beta_or_prob == 'beta':
                label_ylabel = r'Betrouwbaarheidsindex $\beta$ [-/jaar]'
                label_target = 'Doelbetrouwbaarheid'
            elif beta_or_prob =='prob':
                label_ylabel = r'Faalkans $P_f$ [-/jaar]'
                label_target = 'Doelfaalkans'
            labels_xticks = Assessment['Section'].loc[Assessment['index'] == 'Section']
        elif language == 'EN':
            label_xlabel = 'Dike sections'
            if beta_or_prob == 'beta':
                label_ylabel = r'Reliability index $\beta$ [-/year]'
                label_target = 'Target reliability'
            elif beta_or_prob == 'prob':
                label_ylabel = r'Failure probability $P_f$ [-/year]'
                label_target = 'Target failure prob.'
            labels_xticks = []
            for i in Assessment['Section'].loc[Assessment['index'] == 'Section']:
                labels_xticks.append('S' + i[-2:])

        #Derive some coordinates to properly plot everything according to the length of the different sections:
        cumlength = np.cumsum(Assessment['Length'].loc[Assessment['index'] == 'Overflow']).values
        cumlength = np.insert(cumlength, 0, 0)
        xticks1 = copy.deepcopy(cumlength)
        for i in range(1, len(cumlength) - 1):
            xticks1 = np.insert(xticks1, i * 2, cumlength[i])
        middles = (cumlength[:-1] + cumlength[1:]) / 2

        color = ['r', 'g', 'b', 'k']

        for ii in years:
            if first:
                plt.figure(ii, figsize=fig_size)

            plt.figure(ii)
            col = 0

            for j in mechanisms:
                plotdata = Assessment[str(ii)].loc[Assessment['index'] == j].values
                if beta_or_prob == 'prob':
                    plotdata = norm.cdf(-plotdata)

                ydata = copy.deepcopy(plotdata)

                for ij in range(0, len(plotdata)):
                    ydata = np.insert(ydata, ij * 2, plotdata[ij])

                plt.plot(xticks1, ydata, color=color[col], linestyle='-', label=j, alpha=alpha)
                if not first:
                    plt.plot(middles, plotdata, color=color[col], linestyle='', marker='o', alpha=alpha)

                col += 1

            col = 0
            #Whether to draw the target reliability for each individula mechanism.
            if draw_targetbeta == 'on' and last:
                for j in mechanisms:
                    dash = [2, 2]
                    if j == 'StabilityInner':
                        N = self.GeneralInfo['TrajectLength'] * self.GeneralInfo['aStabilityInner']/ self.GeneralInfo['bStabilityInner']
                        pt = self.GeneralInfo['Pmax'] * self.GeneralInfo['omegaStabilityInner']/N
                        # dash = [1,2]
                    elif j == 'Piping':
                        N = self.GeneralInfo['TrajectLength'] * self.GeneralInfo['aPiping']/ self.GeneralInfo['bPiping']
                        pt = self.GeneralInfo['Pmax'] * self.GeneralInfo['omegaPiping'] /N
                        # dash = [1,3]
                    elif j == 'Overflow':
                        pt = self.GeneralInfo['Pmax'] * self.GeneralInfo['omegaOverflow']
                        # dash = [1,2]
                    if beta_or_prob == 'beta':
                        plt.plot([0, max(cumlength)], [-norm.ppf(pt), -norm.ppf(pt)],
                                 color=color[col], linestyle=':', label=label_target + ' ' + j,dashes=dash, alpha=0.5,linewidth=1)
                    elif beta_or_prob == 'prob':
                        plt.plot([0, max(cumlength)], [pt, pt],
                                 color=color[col], linestyle=':', label=label_target + ' ' + j, dashes=dash, alpha=0.5,linewidth=1)
                    col += 1
            if last:
                for i in cumlength:
                    plt.axvline(x=i, color='k', linestyle=':', alpha=0.5)

                if beta_or_prob == 'beta':
                    plt.plot([0, max(cumlength)], [-norm.ppf(self.GeneralInfo['Pmax']), -norm.ppf(self.GeneralInfo['Pmax'])], 'k--', label=label_target, linewidth=1)

                if beta_or_prob == 'prob':
                    plt.plot([0, max(cumlength)], [self.GeneralInfo['Pmax'], self.GeneralInfo['Pmax']], 'k--', label=label_target, linewidth=1)

                plt.legend(loc=1)
                plt.xlabel(label_xlabel)
                plt.ylabel(label_ylabel)
                plt.xticks(middles, labels_xticks, rotation=90)
                plt.xlim([0, max(cumlength)])
                plt.tick_params(axis='both', bottom=False)
                if beta_or_prob == 'beta':
                    plt.ylim([1.5, 8.5])

                if beta_or_prob == 'prob':
                    plt.ylim([1e-1, 1e-9])
                    plt.gca().set_yscale('log')

                plt.grid(axis='y')

                if flip == 'on':
                    plt.gca().invert_xaxis()

                # plt.xlim([np.min(cumlength), np.max(cumlength)])
                if PATH != None:
                    if type == 'Assessment':
                        plt.savefig(PATH.joinpath(beta_or_prob + '_' + str(startyear + ii) + '_Assessment.png'), dpi=300, bbox_inches='tight', format='png')
                    else:
                        plt.savefig(PATH.joinpath(str(startyear + ii) + '_Step=' + str(type) + '_' + beta_or_prob + '.png'), dpi=300, bbox_inches='tight', format='png')
                    plt.close()
                else:
                    plt.show()

    def runFullAssessment(self):
        for i in self.Sections:
            for j in self.GeneralInfo['Mechanisms']:
                i.Reliability.Mechanisms[j].generateLCRProfile(i.Reliability.Load, mechanism=j, trajectinfo=self.GeneralInfo)

            i.Reliability.calcSectionReliability(TrajectInfo=self.GeneralInfo, length=i.Length)

    def plotAssessmentResults(self, directory, section_ids=None, t_start=2020):
        # for all or a selection of sections:
        if section_ids==None:
            sections = self.Sections
        else:
            sections = []
            for i in section_ids:
                sections.append(self.Sections[i])
        createDir(directory)
        #generate plots
        for i in sections:
            plt.figure(1)
            [i.Reliability.Mechanisms[j].drawLCR(label=j, type='Standard', tstart=t_start) for j in self.GeneralInfo['Mechanisms']]
            plt.plot([t_start, t_start + np.max(self.GeneralInfo['T'])],
                     [-norm.ppf(self.GeneralInfo['Pmax']),
                      -norm.ppf(self.GeneralInfo['Pmax'])],
                     'k--', label='Requirement')
            plt.legend()
            plt.title(i.name)
            plt.savefig(directory.joinpath(i.name + '.png'), bbox_inches='tight')
            plt.close()

    def updateProbabilities(self,Probabilities,ChangedSection):
        #This function is to update the probabilities after a reinforcement.
        for i in self.Sections:
            if i.name == ChangedSection:
                i.Reliability.SectionReliability = Probabilities.loc[ChangedSection].astype(float)
        pass
#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.Reliability = SectionReliability()
        self.name = name  #Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
        # Basic traject info NOTE: THIS HAS TO BE REMOVED TO TRAJECT OBJECT
        self.TrajectInfo = {}
        if traject == '16-4':
            self.TrajectInfo['TrajectLength'] = 19480
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
        elif traject == '16-3':
            self.TrajectInfo['TrajectLength'] = 19899
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300

    def readGeneralInfo(self, path, sheet_name):
        #Read general data from sheet in standardized xlsx file
        df = pd.read_excel(path.joinpath(self.name + ".xlsx"), sheet_name=None)

        for name, sheet_data in df.items():
            if name == sheet_name:
                data = df[name].set_index('Name')
                self.MechanismData = {}

                for i in range(len(data)):
                    if data.index[i] == 'Overflow' or data.index[i] == 'Piping' or data.index[i] == 'StabilityInner':
                        self.MechanismData[data.index[i]] = (data.loc[data.index[i]][0], data.loc[data.index[i]][1])
                        # setattr(self, data.index[i], (data.loc[data.index[i]][0], data.loc[data.index[i]][1]))
                    else:
                        setattr(self, data.index[i], (data.loc[data.index[i]][0]))

            elif name == "Housing":
                self.houses = pd.concat([df["Housing"], pd.DataFrame(np.cumsum(df["Housing"]['number'].values), columns=['cumulative'])],axis=1, join='inner').set_index('distancefromtoe')
            else:
                self.houses = None

        #and we add the geometry
        setattr(self, 'InitialGeometry', df['Geometry'])

#Class describing safety assessments of a section:
class SectionReliability:
    def __init__(self):
        self
    def calcSectionReliability(self,TrajectInfo,length = 1000):
        #This routine translates cross-sectional to section reliability indices
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
        beta_mech_time = pd.DataFrame(-norm.ppf(pf_mechanisms_time),
                                          columns=list(self.Mechanisms[list(self.Mechanisms.keys())[0]].Reliability.keys()),
                                          index=list(self.Mechanisms.keys()))
        beta_time = pd.DataFrame([-norm.ppf(np.sum(pf_mechanisms_time,axis=0))],
                         columns=list(self.Mechanisms[list(self.Mechanisms.keys())[0]].Reliability.keys()),
                         index=['Section'])
        self.SectionReliability = pd.concat((beta_mech_time,beta_time))

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
    def __init__(self, mechanism,type,years,measure_year=0):
        #Initialize and make collection of MechanismReliability objects
        #mechanism, type, years are universal.
        # Measure_year is to indicate whether the reliability has to be recalculated or can be copied
        # (the latter is the case if a measure is taken later than the considered point in time)

        self.Reliability = {}

        for i in years:
            if measure_year > i:
                self.Reliability[str(i)] = MechanismReliability(mechanism, type, copy_or_calculate='copy')
            else:
                self.Reliability[str(i)] = MechanismReliability(mechanism, type)

    def generateInputfromDistributions(self, distributions, parameters = ['R', 'dR', 'S'], processes = ['dR']):
        processIDs = []
        for process in processes:
            processIDs.append(parameters.index(process))

        for i in self.Reliability:
            self.Reliability[i].Input.fill_distributions(distributions,np.int32(i),processIDs,parameters)

            pass

    def generateLCRProfile(self, load=False, mechanism='Overflow', method='FORM', trajectinfo=None,
                           conditionality = 'no'):
        # this function generates life-cycle reliability based on the years that have been calculated (so reliability in time)
        if load:
            [self.Reliability[i].calcReliability(self.Reliability[i].Input, load, mechanism=mechanism, method=method, year=float(i), TrajectInfo=trajectinfo) for i in self.Reliability.keys()]
        else:
            [self.Reliability[i].calcReliability(mechanism=mechanism, method=method,
                                                 year=float(i), TrajectInfo=trajectinfo) for i in
             self.Reliability.keys()]
        #NB: This should be extended with conditional failure probabilities

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
        pfs = norm.cdf(-beta)
        pftot = 1 - np.cumprod(1 - pfs)
        self.beta_life = (max(tgrid), np.float(norm.ppf(1 - pftot[-1:])))

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

    def drawLCR(self, yscale=None, type='pf', label=None, tstart=0, newfigure='yes'):
        #Draw the life cycle reliability
        t = []
        y = []

        for i in self.Reliability.keys():
            t.append(float(i)+tstart)
            if self.Reliability[i].type == 'Prob':
                if self.Reliability[i].result.getClassName() == 'SimulationResult':
                    y.append(self.Reliability[i].result.getProbabilityEstimate()) if type == 'pf' else y.append(-ot.Normal().computeScalarQuantile(self.Reliability[i].result.getProbabilityEstimate()))
                else:
                    y.append(self.Reliability[i].result.getEventProbability()) if type == 'pf' else y.append(self.Reliability[i].result.getHasoferReliabilityIndex())
            else:
                y.append(self.Reliability[i].Pf) if type == 'pf' else y.append(self.Reliability[i].beta)

        plt.plot(t, y, label=label)
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
            if i is not 'mechanism':
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
                result_total, Ptotal, wl_total = IterativeFC_calculation(marginals, start, names, Mechanisms.zPipingTotal, method, step, lolim, hilim)
                self.h_c = ot.Distribution(TableDist(np.array(wl_total), np.array(Ptotal), extrap='on'))
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
            self.h_c = ot.Distribution(TableDist(np.array(wl), np.array(P), extrap='on'))
            self.results = result
            self.wl = wl
        elif mechanism == 'StabilityInner':
            pass

        self.type = 'FragilityCurve'

    def calcReliability(self, strength = False, load = False, mechanism=None, method='FORM', year=0, TrajectInfo=None):
        #This routine calculates cross-sectional reliability indices based on different types of calculations.
        if self.type == 'Simple':
            if mechanism == 'StabilityInner':
                #Simple interpolation of two safety factors and translation to a value of beta at 'year'.
                #In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
                SFt = interpolate.interp1d([0, 50],np.concatenate((strength.input['SF_2025']/self.gamma_schem,
                                                                   strength.input['SF_2075']/self.gamma_schem)),
                                                                   fill_value='extrapolate')
                SF = SFt(year)
                modelfactor = 1.07
                beta = np.min([((SF/modelfactor)-0.41)/0.15, 8])
                #Check if there is an elimination measure present (diaphragm wall)
                if 'Elimination' in strength.input.keys():
                    if strength.input['Elimination'] == 'yes':
                        #Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                        self.Pf = norm.cdf(-beta) * strength.input['Pf_elim']  + \
                            strength.input['Pf_with_elim'] * (1-strength.input['Pf_elim'])
                        self.beta = -norm.ppf(self.Pf)
                    else:
                        raise ValueError('Warning: Elimination defined but not turned on')
                else:
                    self.beta = beta
                    self.Pf = norm.cdf(-self.beta)

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
                marginals = [self.h_c, load.distribution, dist_change]
                dist = ot.ComposedDistribution(marginals)
                dist.setDescription(['h_c', 'h', 'dh'])
                result, P, beta, alfas_sq = run_prob_calc(ot.SymbolicFunction(['h_c', 'h', 'dh'], ['h_c-(h+dh)']), dist,
                                                          method)
            else:
                marginals = [self.h_c, load.distribution]
                dist = ot.ComposedDistribution(marginals)
                dist.setDescription(['h_c', 'h'])
                result, P, beta, alfas_sq = run_prob_calc(ot.SymbolicFunction(['h_c', 'h'], ['h_c-h']), dist, method)
            self.result = result
            self.Pf = P
            self.beta = beta
            self.alpha_sq = alfas_sq
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

                inputs = addLoadCharVals(strength_new.input, load=load, p_h=TrajectInfo['Pmax'], p_dh=0.5, year=year)
                Z, self.p_dh, self.p_dh_c = Mechanisms.zPiping(inputs, mode='SemiProb')
                self.gamma_pip = ProbabilisticFunctions.calc_gamma('Piping', TrajectInfo=TrajectInfo) # Calculate needed safety factor

                self.SF_p = (self.p_dh_c / (self.gamma_pip * self.gamma_schem_pip)) / self.p_dh
                self.assess_p = 'voldoende' if self.SF_p > 1 else 'onvoldoende'
                self.beta_cs_p = ProbabilisticFunctions.calc_beta_implicated('Piping', (self.p_dh_c/self.gamma_schem_pip) / self.p_dh,TrajectInfo=TrajectInfo)  # Calculate the implicated beta_cs

                # Heave:
                Z, self.h_i, self.h_i_c = Mechanisms.zHeave(inputs,mode='SemiProb')
                self.gamma_h = ProbabilisticFunctions.calc_gamma('Heave',TrajectInfo=TrajectInfo)  # Calculate needed safety factor

                self.SF_h = (self.h_i_c / (self.gamma_schem_heave * self.gamma_h)) / self.h_i
                self.assess_h = 'voldoende' if (self.h_i_c / (self.gamma_schem_heave * self.gamma_h)) / self.h_i > 1 else 'onvoldoende'
                self.beta_cs_h = ProbabilisticFunctions.calc_beta_implicated('Heave', (self.h_i_c/self.gamma_schem_heave) / self.h_i,TrajectInfo=TrajectInfo)  # Calculate the implicated beta_cs

                # Uplift
                Z, self.u_dh, self.u_dh_c = Mechanisms.zUplift(inputs,mode='SemiProb')
                self.gamma_u = ProbabilisticFunctions.calc_gamma('Uplift',TrajectInfo=TrajectInfo)  # Calculate needed safety factor

                self.SF_u = (self.u_dh_c / (self.gamma_schem_upl * self.gamma_u)) / self.u_dh
                self.assess_u = 'voldoende' if (self.u_dh_c / (self.gamma_schem_upl * self.gamma_u)) / self.u_dh > 1 else 'onvoldoende'
                self.beta_cs_u = ProbabilisticFunctions.calc_beta_implicated('Uplift', (self.u_dh_c/self.gamma_schem_upl) / self.u_dh,TrajectInfo=TrajectInfo)  # Calculate the implicated beta_cs

                #Check if there is an elimination measure present (VZG or diaphragm wall)
                if 'Elimination' in strength.input.keys():
                    if strength.input['Elimination'] == 'yes':
                        #Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                        self.Pf = \
                            norm.cdf(-np.max([self.beta_cs_h,self.beta_cs_u,self.beta_cs_p])) * strength.input['Pf_elim']  + \
                            strength.input['Pf_with_elim'] * (1-strength.input['Pf_elim'])
                        self.beta = -norm.ppf(self.Pf)
                    else:
                        raise ValueError('Warning: Elimination defined but not turned on')
                else:
                    self.beta = np.min([np.max([self.beta_cs_h,self.beta_cs_u,self.beta_cs_p]),8])
                    self.Pf = norm.cdf(-self.beta)
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
            if calctype == 'Prob':
                #if it is a probabilistic calculation, transform the data to OpenTurns distributions:
                if data.iloc[i]['Distribution'] == 'L':                 #Lognormal distribution
                    data.at[data.index[i],'Par2'] = data.at[data.index[i],'Par2']*data.at[data.index[i],'Par1'] if data.at[data.index[i],'variance type'] == 'var' else data.at[data.index[i],'Par2']
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
                self.char_vals[data.index[i]] = np.array(self.input[data.index[i]].computeQuantile(data.iloc[i]['quantile']))[0]
                self.temporals.append(data.index[i]) if data.iloc[i]['temporal'] == 'yes' else self.temporals
            else:
                x = data.iloc[i][:].values
                x = x.astype(np.float32)
                self.input[data.index[i]] = x[~np.isnan(x)]
                if data.index[i][-3:] == '(t)':
                    self.temporals.append(data.index[i])

#routine to extract a characteristic 4 or 6 point profile from a list of xz coordinates:
def extractProfile(profile,window=5,titel='Profile',path = None):
    profile_averaged = copy.deepcopy(profile)
    profile_averaged['z'] = pd.rolling_mean(profile['z'], window=window, center=True, min_periods=1)

    profile_cov = copy.deepcopy(profile_averaged)
    profile_cov['z'] = pd.rolling_cov(profile['z'], window=window, center=True, min_periods=1)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(profile['x'], profile['z'])
    ax1.plot(profile_averaged['x'], profile_averaged['z'], 'r')

    ax2.plot(profile_cov['x'], profile_cov['z'])


    d1_peaks = np.argwhere(profile_cov['z'] > 0.02).flatten()
    ax1.plot(profile_averaged['x'].iloc[d1_peaks], profile_averaged['z'].iloc[d1_peaks], 'or')
    d1_peaks_diff = np.diff(d1_peaks)
    x = []
    takenext = 0

    #This finds the crest:
    for i in range(0, len(d1_peaks_diff)):
        if i == 0 or d1_peaks_diff[i] > 3 or takenext == 1:
            if takenext == 1:
                if np.absolute([d1_peaks[i]- d1_peaks[i - 1]]) < 20 and np.absolute([d1_peaks[i]- d1_peaks[i - 1]]) > 3:
                    x.append(d1_peaks[i])
                    takenext = 0
                else:
                    takenext = 0
            else:
                x.append(d1_peaks[i])
                takenext = 1
    # ax2.plot(profile_averaged['x'].iloc[x], np.ones((len(x), 1)), 'xb')
    # for i in x:
    #     ax1.axvline(x=profile_averaged['x'].iloc[i], LineStyle='--', Color='k')
    # if len(x) > 6:
    #     #selecteer de juiste punten
    # elif len (x) < 4:
    #     #not enough points found
    # else:
    #select two highest points to find crest
    #crest:
    crest = profile_averaged.iloc[x].nlargest(2,'z')
    ind_crest = crest.index.values
    x_crest = crest['x'].values
    z_crest = np.average(crest['z'].values)

    inwardside = profile_averaged.iloc[0:np.min(ind_crest)]
    toe_in_est_idx = np.int(np.round(3 * z_crest / 0.5))
    mean_in = np.median(inwardside['z'].iloc[-50-toe_in_est_idx:-toe_in_est_idx])
    inwardside = inwardside.iloc[-3 * toe_in_est_idx:]
    inward_cov = pd.rolling_cov(inwardside['z'], window=window, center=True, min_periods=1)

    inward_crossing = inwardside.loc[inward_cov > 0.02]
    inward_crossing = inward_crossing.loc[inward_crossing['z'] < np.max([mean_in + 0.3, np.min(inward_crossing['z'])+0.01])]

    x_inner = inward_crossing['x'].iloc[-1]
    z_inner = inward_crossing['z'].iloc[-1]
    ind_inner = inward_crossing.loc[inward_crossing['z'] == z_inner].index.values[0]

    outwardside = profile_averaged.iloc[np.max(ind_crest):]
    mean_out = np.median(outwardside['z'].iloc[0:75])
    outward_crossing = outwardside.loc[outwardside['z'] < mean_out+0.1]
    x_outer = outward_crossing['x'].values[0]
    z_outer = outward_crossing['z'].values[0]

    #berm
    #2 points between crest and toe
    berm = profile_averaged.iloc[ind_inner:np.min(ind_crest)]
    berm_cov = pd.rolling_cov(berm['z'], window=window, center=True, min_periods=1)
    berm_points = berm.loc[berm_cov < 0.05]
    berm_points = berm_points.loc[berm_points['z'] > mean_in +1.5]


    if len(berm_points) > 1:
        berm_points = berm_points.iloc[[0,-1]]
        x_berm = berm_points['x'].values
        z_berm = np.average(berm_points['z'].values)
        #check if berm makes sense by verifying if slope is about ok:
        # slope lower part  and upper part should both be > 1.5
        if (np.min(x_berm)-x_inner)/(z_berm-z_inner) > 1.5 and (np.min(x_crest)-np.max(x_berm))/(z_crest-z_berm) > 1.5 and np.diff(x_berm) > 2:
            x_values = np.array([x_inner, x_berm[0], x_berm[1], np.min(x_crest), np.max(x_crest), x_outer])
            z_values = np.array([z_inner, z_berm, z_berm, z_crest, z_crest, z_outer])
        else:
            x_values = np.array([x_inner, np.min(x_crest), np.max(x_crest), x_outer])
            z_values = np.array([z_inner, z_crest, z_crest, z_outer])
            print('For ' + titel + ' estimated berm was deleted')
            print('lower slope: ' + str((np.min(x_berm)-x_inner)/(z_berm-z_inner)))
            print('upper slope: ' + str((np.min(x_crest)-np.max(x_berm))/(z_crest-z_berm)))
            print('berm length: ' + str(np.diff(x_berm)))
        #add berm
        # last step
        # filter out bogus berms where slopes are way too steep


    else:
        x_values = np.array([x_inner, np.min(x_crest), np.max(x_crest), x_outer])
        z_values = np.array([z_inner, z_crest, z_crest, z_outer])
        #not
    x_values.flatten()
    z_values.flatten()

    #adapt points that were not good:
    if titel =='VY094': x_values[0] = 122; z_values[0] = 2.5; print('Adapted inner toe for VY094')
    if titel =='VY058': x_values[0] = 131.5; z_values[0]  = 3.; x_values[-1] = 158;  z_values[-1] = 4.8; print('Adapted inner and outer toe for VY058')
    if titel =='AW216': x_values[1] = 139; z_values[1] = 5.5; x_values[2] = 149.6; z_values[2] = 5.5; print('Adapted crest for AW216')
    if titel == 'AW219':
        x_values[1] = 137.6; z_values[1] = 5.9; x_values[2] = 149; z_values[2] = 5.9; print('Corrected crest of AW219')
    if titel == 'AW240': x_values[0] = 124; z_values[0] = 1.07; print('Corrected inner toe of AW240')
    if titel == 'AW248': x_values[-1] = 152; z_values[-1] = 2.12; print('Corrected outer toe of AW248')
    if titel == 'AW276':
        x_values = np.insert(x_values,[0, 1], [100, 135])
        z_values = np.insert(z_values,[0, 1], [-0.7, z_values[0]])
        print('Corrected AW276')

    # Plot the  points with a line
    ax1.plot(x_values,z_values, color = 'k', marker = 'o', linestyle = '--')
    ax1.set_ylabel('m NAP')
    ax2.set_ylabel('CoV profiel')
    ax1.set_xlim(np.min(x_values)-30, np.max(x_values)+30)
    ax2.set_xlim(np.min(x_values)-30, np.max(x_values)+30)
    fig.suptitle(titel)
    if path != None:
        plt.savefig(path + '\\' + titel + '.png')
        plt.close()


    return pd.DataFrame(np.hstack((x_values[:,None], z_values[:,None])),columns=['x','z'])




    #maximaal 4 punten links van de bkl op 150 m. max 1 punt rechts daarvan
    # y coordinaten:
    # teen is y coordinaat van het punt zelf uit gladgestreken profiel
    # bkl en binnenkl gemiddelde van de twee, of de kruinhoogte uit Overflow (later in te voeren dan)
    # bermhoogte gemiddelde van de twee lijnen
    # binnenteen: punt zelf uit gladgestreken profiel
    #     #
    # from shapely.geometry import Polygon
    # # topolygonpoints, skip first and last 20 points
    # pointss = []
    # for i in range(20, len(profile_averaged) - 20):
    #     pointss.append((profile_averaged['x'].iloc[i], profile_averaged['z'].iloc[i]))
    # pointss.append((profile_averaged['x'].iloc[len(profile_averaged) - 19], np.min(profile_averaged['z'])))
    # polygonshape = Polygon(pointss)
    # polygonshape = polygonshape.simplify(0.1)
    # x1, y1 = polygonshape.exterior.xy
    # ax1.fill(x1, y1, 'g', alpha=0.5)

#DEPENDENT IMPORT (this has to be here to prevent a circular reference in the code)
from HelperFunctions import createDir
