import copy
import matplotlib.pyplot as plt
import numpy as np
from DikeSection import DikeSection
from ReliabilityCalculation import LoadInput, MechanismReliabilityCollection
import ProbabilisticFunctions
from StrategyEvaluation import calcTrajectProb
import time
import seaborn as sns

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
                                                                                     'Piping'],makesubdirs=True):
        #Make a case directory and inside a figures and results directory if it doesnt exist yet
        if not path.joinpath(directory).is_dir():
            path.joinpath(directory).mkdir(parents=True, exist_ok=True)
            if makesubdirs:
                path.joinpath(directory).joinpath('figures').mkdir(parents=True, exist_ok=True)
                path.joinpath(directory).joinpath('results', 'investment_steps').mkdir(parents=True, exist_ok=True)

        # Routine to read the input for all sections based on the default input format.
        files = [i for i in path.glob("*DV*") if i.is_file()]
        if len(files) == 0:
            raise IOError('Error: no dike sections found. Check path!')
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
            if not path.joinpath(directory).joinpath('figures', self.Sections[i].name).is_dir() and makesubdirs:
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
        Assessment = Assessment.rename(columns={'index':'mechanism','Section':'name'})
        self.Probabilities = Assessment.reset_index(drop=True).set_index(['name','mechanism'])

    def plotAssessment(self,
                       PATH = None, fig_size = (6,4),
                       draw_targetbeta='off', language='EN',
                       flip='off', beta_or_prob = 'beta',
                       outputcsv=False, last=True, alpha = 1, years = False, colors=False,
                       labels_limited=False,system_rel=False, custom_name=False,title_in=False,
                       reinforcement_strategy=False,greedymode='Optimal',show_xticks = True):
        '''Routine to plot traject reliability'''
        if reinforcement_strategy:
            if reinforcement_strategy.__class__.__name__ == 'GreedyStrategy':
                if greedymode == 'Optimal':
                    ProbabilityFrame = reinforcement_strategy.Probabilities[reinforcement_strategy.OptimalStep]
                elif greedymode == 'SafetyStandard':
                    Ptarget = self.GeneralInfo['Pmax']
                    for i in reversed(reinforcement_strategy.Probabilities):
                        beta_traj, Pf_traj = calcTrajectProb(i, ts=50)
                        if Pf_traj < Ptarget:  # satisfactory solution
                            ProbabilityFrame = i
                        else:
                            if not 'ProbabilityFrame' in locals(): raise ValueError('No satisfactory solution found')
                            break
            else:
                ProbabilityFrame = reinforcement_strategy.Probabilities[-1]
        else:
            ProbabilityFrame = self.Probabilities
            ProbabilityFrame = ProbabilityFrame.drop(['Length'],axis=1)
        ProbabilityFrame.columns = ProbabilityFrame.columns.values.astype(np.int64)
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
                if labels_limited:
                    label_ylabel = r'$\beta$ [-/year]'
                else:
                    label_ylabel = r'Reliability index $\beta$ [-/year]'
                label_target = r'$\beta_\mathrm{target}$'
            elif beta_or_prob == 'prob':
                label_ylabel = r'Failure probability $P_f$ [-/year]'
                label_target = 'Target failure prob.'
            labels_xticks = []
            for i in self.Sections:
                labels_xticks.append('S' + i.name[-2:])

        cumlength, xticks1, middles = getSectionLengthInTraject(self.Probabilities['Length'].loc[self.Probabilities.index.get_level_values(1) == 'Overflow'].values)

        # color = ['r', 'g', 'b', 'k']
        if colors:
            color = sns.cubehelix_palette(**colors)
        else:
            color = sns.cubehelix_palette(n_colors=4, start=1.9,rot=1,gamma=1.5,hue=1.0,light=0.8,dark=0.3)
        # color = sns.cubehelix_palette(n_colors=4, start=0.7,rot=1,gamma=1.5,hue=0.0,light=0.8,dark=0.3)
        markers = ['o', 'v', 'd']
        # fig, ax = plt.subplots(figsize=fig_size)
        #We will make many plots for different years
        year = 0
        line = {}
        mid = {}
        legend_line = {}
        for ii in years:
            if system_rel:
                fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=fig_size,sharey='row',
                                              gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.08, 'left': 0.03, 'right': 0.98})
            else:
                fig, ax = plt.subplots(figsize=fig_size)
            col = 0
            mech = 0
            for j in mechanisms:
                #get data to plot
                # plotdata = self.Probabilities[str(ii)].loc[self.Probabilities['index'] == j].values
                plotdata = ProbabilityFrame[ii].loc[ProbabilityFrame.index.get_level_values(1) == j].values
                if beta_or_prob == 'prob':
                    plotdata = ProbabilisticFunctions.beta_to_pf(plotdata)
                ydata = copy.deepcopy(plotdata)
                for ij in range(0, len(plotdata)):
                    ydata = np.insert(ydata, ij * 2, plotdata[ij])

                if  year < 1000: #year == 0:
                    #define the lines for the first time. Else replace the data.
                    line[mech], =  ax.plot(xticks1, ydata, color=color[col], linestyle='-', alpha=alpha)
                    mid[mech], = ax.plot(middles, plotdata, color=color[col], linestyle='', marker=markers[col], alpha=alpha)
                    legend_line[mech], = ax.plot(-999, -999, color=color[col], linestyle='-', marker=markers[col], alpha=alpha, label=j )
                else:
                    line[mech].set_ydata(ydata)
                    mid[mech].set_ydata(plotdata)
                col += 1
                mech += 1
            if system_rel: legend_line[mech], = ax.plot(-999, -999, color=color[col], linestyle='-', alpha=alpha, label='System')
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
                        ax.plot([0, max(cumlength)], [ProbabilisticFunctions.pf_to_beta(pt), ProbabilisticFunctions.pf_to_beta(pt)],
                                 color=color[col], linestyle=':', label=label_target + ' ' + j,dashes=dash, alpha=0.5,linewidth=1)
                    elif beta_or_prob == 'prob':
                        ax.plot([0, max(cumlength)], [pt, pt],
                                 color=color[col], linestyle=':', label=label_target + ' ' + j, dashes=dash, alpha=0.5,linewidth=1)
                    col += 1
            if last:
                for i in cumlength:
                    ax.axvline(x=i, color='gray', linestyle='-',linewidth=0.5, alpha=0.5)
                if beta_or_prob == 'beta':
                    #should be in legend
                    ax.plot([-100,-99], [ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(self.GeneralInfo[
                                                                                                    'Pmax'])], 'k--', label=label_target, linewidth=1)
                    #visible in bar chart
                    if system_rel: ax1.axhline(ProbabilisticFunctions.pf_to_beta(self.GeneralInfo['Pmax']), linestyle='--',color='black', label=label_target, linewidth=1)
                if beta_or_prob == 'prob':
                    ax.plot([0, max(cumlength)], [self.GeneralInfo['Pmax'], self.GeneralInfo['Pmax']], 'k--',
                           label=label_target, linewidth=1)

                ax.legend(loc=1)
                if not labels_limited: ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                if show_xticks:
                    ax.set_xticklabels(labels_xticks)
                else:
                    ax.set_xticklabels('')
                ax.tick_params(axis='x',rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis='both', bottom=False)
                if beta_or_prob == 'beta':
                    ax.set_ylim([0.5, 8.5])

                if beta_or_prob == 'prob':
                    ax.set_ylim([1e-1, 1e-9])
                    ax.set_yscale('log')

                ax.grid(axis='y',linewidth=0.5,color='gray',alpha=0.5)

                if flip == 'on':
                    ax.invert_xaxis()
            if system_rel:
                col=0
                mech=0
                line1 = {}
                mid1 = {}
                bars = {}
                pt_tot = 0
                for m in mechanisms:
                    beta_t,p_t = calcTrajectProb(ProbabilityFrame,ts=ii,mechs=[m])
                    # pt_tot +=p_t
                    pt_tot = 1-((1-pt_tot)*(1-p_t))
                    # line1[mech], = ax1.plot([0,1], [beta_t,beta_t], color=color[col], linestyle='-', label=j, alpha=alpha)
                    # mid1[mech], = ax1.plot(0.5, beta_t, color=color[col], linestyle='', marker=markers[col], alpha=alpha)
                    bars[mech] = ax1.bar(col,beta_t,color=color[col])
                    col += 1
                    mech += 1
                beta_tot = ProbabilisticFunctions.pf_to_beta(pt_tot)
                print(beta_tot)
                ax1.plot([-2,3],[beta_tot, beta_tot], color=color[col])
                ax1.grid(axis='y',linewidth=0.5,color='gray',alpha=0.5)
                ax1.set_xticks([0,1,2])
                ax1.set_xlim(left=-0.4,right=2.4)
                if show_xticks:
                    ax1.set_xticklabels(mechanisms,rotation=90,fontsize=6)
                else:
                    ax1.set_xticklabels('')
                ax1.tick_params(axis='both', bottom=False)
                if title_in:
                    ax.set_title(title_in)
            if PATH != None:
                if not custom_name: custom_name = beta_or_prob + '_' + str(startyear + ii) + '_Assessment.png'
                plt.savefig(PATH.joinpath(custom_name), dpi=300, bbox_inches='tight', format='png')
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