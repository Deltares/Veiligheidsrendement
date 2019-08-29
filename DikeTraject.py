import copy
import matplotlib.pyplot as plt
import numpy as np
from DikeSection import DikeSection
from ReliabilityCalculation import LoadInput, MechanismReliabilityCollection
import ProbabilisticFunctions
import time

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
        elif traject == '16-3 en 16-4':
            self.GeneralInfo['FloodDamage'] = 23e9
            self.GeneralInfo['TrajectLength'] = 19500 #voor doorsnede-eisen wel ongeveer lengte individueel traject
            # gebruiken
            self.GeneralInfo['Pmax'] = 1. / 10000
            self.GeneralInfo['omegaPiping'] = 0.24; self.GeneralInfo['aPiping'] = 0.9; self.GeneralInfo['bPiping'] = 300
            self.GeneralInfo['omegaStabilityInner'] = 0.04; self.GeneralInfo['aStabilityInner'] = 0.033; self.GeneralInfo['bStabilityInner'] = 50
            self.GeneralInfo['omegaOverflow'] = 0.24
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

        self.GeneralInfo['beta_max'] = ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax'])
        self.GeneralInfo['gammaHeave'] = ProbabilisticFunctions.calc_gamma('Heave',self.GeneralInfo)
        self.GeneralInfo['gammaUplift'] = ProbabilisticFunctions.calc_gamma('Uplift',self.GeneralInfo)
        self.GeneralInfo['gammaPiping'] = ProbabilisticFunctions.calc_gamma('Piping',self.GeneralInfo)

    def setProbabilities(self):
        '''routine to make 1 dataframe of all probabilities of a TrajectObject'''
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
        self.Probabilities = Assessment.reset_index(drop=True)

    def plotAssessment(self,
                       PATH = None, fig_size = (6,4),
                       draw_targetbeta='off', language='EN',
                       flip='off', beta_or_prob = 'beta',
                       outputcsv=False, last=True, alpha = 1, years = False):
        '''Routine to plot traject reliability'''
        PlotSettings()
        if not years:
            years = self.GeneralInfo['T']
        mechanisms = self.GeneralInfo['Mechanisms']
        startyear = self.GeneralInfo['StartYear']
        if outputcsv:
            self.Probabilities.to_csv(PATH.joinpath('AllBetas.csv'))
        #English or Dutch labels and titles
        if language == 'NL':
            label_xlabel = 'Dijkvakken'
            if beta_or_prob == 'beta':
                label_ylabel = r'Betrouwbaarheidsindex $\beta$ [-/jaar]'
                label_target = 'Doelbetrouwbaarheid'
            elif beta_or_prob =='prob':
                label_ylabel = r'Faalkans $P_f$ [-/jaar]'
                label_target = 'Doelfaalkans'
            labels_xticks = []
            for i in self.Sections:
                labels_xticks.append(i.name)
        elif language == 'EN':
            label_xlabel = 'Dike sections'
            if beta_or_prob == 'beta':
                label_ylabel = r'Reliability index $\beta$ [-/year]'
                label_target = 'Target reliability'
            elif beta_or_prob == 'prob':
                label_ylabel = r'Failure probability $P_f$ [-/year]'
                label_target = 'Target failure prob.'
            labels_xticks = []
            for i in self.Sections:
                labels_xticks.append('S' + i.name[-2:])

        cumlength, xticks1, middles = getSectionLengthInTraject(self.Probabilities['Length'].loc[self.Probabilities[
                                                                                                 'index']
                                                                                       =='Overflow'].values)

        color = ['r', 'g', 'b', 'k']

        # fig, ax = plt.subplots(figsize=fig_size)
        #We will make many plots for different years
        year = 0
        line = {}
        mid = {}
        for ii in years:
            fig, ax = plt.subplots(figsize=fig_size)
            col = 0
            mech = 0
            for j in mechanisms:
                #get data to plot
                plotdata = self.Probabilities[str(ii)].loc[self.Probabilities['index'] == j].values
                if beta_or_prob == 'prob':
                    plotdata = ProbabilisticFunctions.beta_to_pf(plotdata)
                ydata = copy.deepcopy(plotdata)
                for ij in range(0, len(plotdata)):
                    ydata = np.insert(ydata, ij * 2, plotdata[ij])

                if  year < 1000: #year == 0:
                    #define the lines for the first time. Else replace the data.
                    line[mech], =  ax.plot(xticks1, ydata, color=color[col], linestyle='-', label=j, alpha=alpha)
                    mid[mech], = ax.plot(middles, plotdata, color=color[col], linestyle='', marker='o', alpha=alpha)
                else:
                    line[mech].set_ydata(ydata)
                    mid[mech].set_ydata(plotdata)
                col += 1
                mech += 1
            col = 0
            ## TODO fix this plot routine which was supposed to make everything much faster
            if year < 1000: #year == 0:
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
                            ax.plot([0, max(cumlength)], [ProbabilisticFunctions.pf_to_beta(pt), ProbabilisticFunctions.pf_to_beta(pt)],
                                     color=color[col], linestyle=':', label=label_target + ' ' + j,dashes=dash, alpha=0.5,linewidth=1)
                        elif beta_or_prob == 'prob':
                            ax.plot([0, max(cumlength)], [pt, pt],
                                     color=color[col], linestyle=':', label=label_target + ' ' + j, dashes=dash, alpha=0.5,linewidth=1)
                        col += 1
                if last:
                    for i in cumlength:
                        ax.axvline(x=i, color='k', linestyle=':', alpha=0.5)
                    if beta_or_prob == 'beta':
                        ax.plot([0, max(cumlength)], [ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(self.GeneralInfo[
                                                                                                        'Pmax'])], 'k--', label=label_target, linewidth=1)
                    if beta_or_prob == 'prob':
                        ax.plot([0, max(cumlength)], [self.GeneralInfo['Pmax'], self.GeneralInfo['Pmax']], 'k--',
                               label=label_target, linewidth=1)

                    ax.legend(loc=1)
                    ax.set_xlabel(label_xlabel)
                    ax.set_ylabel(label_ylabel)
                    ax.set_xticks(middles)
                    ax.set_xticklabels(labels_xticks)
                    ax.tick_params(axis='x',rotation=90)
                    ax.set_xlim([0, max(cumlength)])
                    ax.tick_params(axis='both', bottom=False)
                    if beta_or_prob == 'beta':
                        ax.set_ylim([1.5, 8.5])

                    if beta_or_prob == 'prob':
                        ax.set_ylim([1e-1, 1e-9])
                        ax.set_yscale('log')

                    ax.grid(axis='y')

                    if flip == 'on':
                        ax.invert_xaxis()
            else: #inactive loop
                for i in range(0,mech):
                    ax.draw_artist(line[i])
                    ax.draw_artist(mid[i])
                fig.canvas.update()
                fig.canvas.flush_events()
            if PATH != None:
                plt.savefig(PATH.joinpath(beta_or_prob + '_' + str(startyear + ii) + '_Assessment.png'), dpi=300, bbox_inches='tight', format='png')
                plt.close()
            else:
                pass
            year += 1
            if last:
                plt.close()
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
                     [ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax']),
                      ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax'])],
                     'k--', label='Requirement')
            plt.legend()
            plt.title(i.name)
            plt.savefig(directory.joinpath(i.name + '.png'), bbox_inches='tight')
            plt.close()

    def updateProbabilities(self,Probabilities,ChangedSection=False):
        #This function is to update the probabilities after a reinforcement.
        for i in self.Sections:
            if (ChangedSection) and (i.name == ChangedSection):
                i.Reliability.SectionReliability = Probabilities.loc[ChangedSection].astype(float)
            elif not ChangedSection:
                i.Reliability.SectionReliability = Probabilities.loc[i.name].astype(float)
        pass

def PlotSettings(labels = 'NL'):
    # a bunch of settings to make it look nice:
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

def getSectionLengthInTraject(length):
    # Derive some coordinates to properly plot everything according to the length of the different sections:
    cumlength = np.cumsum(length)
    cumlength = np.insert(cumlength, 0, 0)
    xticks1 = copy.deepcopy(cumlength)
    for i in range(1, len(cumlength) - 1):
        xticks1 = np.insert(xticks1, i * 2, cumlength[i])
    middles = (cumlength[:-1] + cumlength[1:]) / 2
    return cumlength, xticks1, middles

#DEPENDENT IMPORT (this has to be here to prevent a circular reference in the code)
from HelperFunctions import createDir