from collections import OrderedDict
import cProfile
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cplex
import itertools
import ProbabilisticFunctions
from HelperFunctions import IDtoName, flatten, pareto_frontier
import time
from StrategyEvaluation import MeasureCombinations, makeTrajectDF, calcTC, calcTR, calcLifeCycleRisks, \
    calcTrajectProb, ImplementOption, split_options, SolveMIP, getTrajectProb,evaluateRisk, updateProbability
from DikeTraject import PlotSettings, getSectionLengthInTraject
import matplotlib.pyplot as plt


class Strategy:
    #define a strategy object to evaluate measures in accordance with a strategy: options are:
    # TC/Greedy
    # OI/TargetReliability
    # MultiInteger (under dev.)
    # GeneticAlgorithm (under dev.)
    def __init__(self, type, r=0.03):
        self.type = type        #OI or CB
        self.r = r

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

    def evaluate(self, traject, solutions, OI_horizon=50, OI_year=0, splitparams = False,setting='fast'):
        #This is the core code of the optimization. This piece should probably be split into the different methods available.

        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # measures at t=0 (2025) and t=20 (2045)

        if self.type == 'SmartOI' or self.type == 'SmartTargetReliability':
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

    def make_optimization_input(self, traject,solutions):
        # TODO Currently measures with sh = 0.5 crest and sg 0.5 crest + geotextile have not cost 1e99. However they
        #  do have costs higher than the correct option (sh=0m, sg=0.5+VZG) so they will never be selected. This
        #  should be fixed though

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
                self.Pf[i][n,0:np.size(betas[i],0),:]      = ProbabilisticFunctions.beta_to_pf(betas[i])


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
                        if (self.options_geotechnical[keys[n]].iloc[sg]['type'].values[0] != self.options_height[keys[
                            n]].iloc[sh]['type'].values[0]) or (self.options_geotechnical[keys[n]].iloc[sg]['year'].values[0] !=self.options_height[keys[n]].iloc[sh]['year'].values[0]):
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
                pf1 = ProbabilisticFunctions.beta_to_pf(self.options[i]['Section'])
                pftot1 = interp1d(tgrid, pf1)
                risk1 = np.sum(pftot1(np.arange(0, horizon, 1)) * (damage / (1 + r) ** np.arange(0, horizon, 1)),axis=1)
                paretolcc,paretorisk,index1 = pareto_frontier(LCC,risk1,maxX=False,maxY=False)
                index= index1

                # pf1 = ProbabilisticFunctions.beta_to_pf(self.options[i]['StabilityInner']) + ProbabilisticFunctions.beta_to_pf(self.options[i]['Piping'])
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

    def makeFinalSolution(self, path):
        pass
        AllMeasures = copy.deepcopy(self.TakenMeasures)
        # get unique section names
        # sort dataframe by section name (exclude first row)
        # Loop over ordered section names
        # for each section: sum the cost, erase all but the last measure, set BC to Nan, and put LCC in it
        AllMeasures = copy.deepcopy(self.TakenMeasures)
        sections = np.unique(AllMeasures['Section'][1:])
        self.FinalSolution = pd.DataFrame(columns=AllMeasures.columns)
        for section in sections:
            lines = AllMeasures.loc[AllMeasures['Section'] == section]
            if len(lines) > 1:
                lcctot = np.sum(lines['LCC'])
                lines.iloc[-1:]['LCC'] = lcctot
                lines.iloc[-1:]['BC'] = np.nan
                self.FinalSolution = pd.concat([self.FinalSolution, lines[-1:]])
            else:
                self.FinalSolution = pd.concat([self.FinalSolution, lines])
                self.FinalSolution.iloc[-1:]['BC'] = np.nan

        self.FinalSolution.to_csv(path)

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
            plt.plot([2025, 2025+horizon], [ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax'])],
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

            plt.plot([0, np.max(Costs)], [ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax'])], 'k--', label='Norm')
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
                    plt.plot([0, ceiling], [ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(Traject.GeneralInfo['Pmax'])], 'k--', label='Norm')

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

    def plotInvestmentSteps(self, TestCase, investmentlimit = False, step2 = False, path=None, figure_size=(6, 4),
                            years = [0],
                            language='NL', flip=False, alpha=0.3):
        #all settings, similar to plotAssessment

        ##Settings part:
        PlotSettings()
        mechanisms = TestCase.GeneralInfo['Mechanisms']
        startyear = TestCase.GeneralInfo['StartYear']
        #English or Dutch labels and titles
        if language == 'NL':
            label_xlabel = 'Dijkvakken'
            label_ylabel = r'Betrouwbaarheidsindex $\beta$ [-/jaar]'
            label_target = 'Doelbetrouwbaarheid'
            labels_xticks = []
            for i in TestCase.Sections:
                labels_xticks.append(i.name)
        elif language == 'EN':
            label_xlabel = 'Dike sections'
            label_ylabel = r'Reliability index $\beta$ [-/year]'
            label_target = 'Target reliability'
            labels_xticks = []
            for i in TestCase.Sections:
                labels_xticks.append('S' + i[-2:])
        color = ['r', 'g', 'b', 'k']

        cumlength, xticks1, middles = getSectionLengthInTraject(TestCase.Probabilities['Length'].loc[
                                                                    TestCase.Probabilities['index'] ==
                                                                    'Overflow'].values)
        if not investmentlimit:
            # plot the probabilities for i-1 with alpha 0.3
            for i in range(1, len(self.TakenMeasures)):
                col = 0; mech = 0; line1 = {}; line2 = {}; mid = {}
                fig, ax = plt.subplots(figsize = figure_size)
                for j in mechanisms:
                    plotdata1 = copy.deepcopy(self.Probabilities[i-1][years[0]].xs(j,level='mechanism',axis=0)).values
                    plotdata2 = copy.deepcopy(self.Probabilities[i][years[0]].xs(j,level='mechanism',axis=0)).values
                    ydata1 = copy.deepcopy(plotdata1) #oude sterkte
                    ydata2 = copy.deepcopy(plotdata2) #nieuwe sterkte
                    for ij in range(0,len(plotdata1)):
                        ydata1 = np.insert(ydata1, ij*2, plotdata1[ij])
                        ydata2 = np.insert(ydata2, ij*2, plotdata2[ij])
                    line1[mech], = ax.plot(xticks1,ydata1, color=color[col], linestyle='-', alpha=alpha)
                    line2[mech], = ax.plot(xticks1,ydata2, color=color[col], linestyle='-', label=j)
                    mid[mech],   = ax.plot(middles,plotdata2, color = color[col], linestyle = '', marker = 'o')
                    col += 1
                    mech += 1
                for ik in cumlength: ax.axvline(x=ik, color='k', linestyle=':', alpha=0.5)
                ax.plot([0, max(cumlength)], [ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax']),
                                              ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax'])],
                                              'k--', label=label_target, linewidth=1)
                ax.legend(loc=1)
                ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                ax.set_xticklabels(labels_xticks)
                ax.tick_params(axis='x', rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis='both', bottom=False)
                ax.set_ylim([1.5, 8.5])
                ax.grid(axis='y')
                if flip: ax.invert_xaxis()
                plt.savefig(path.joinpath(str(2025+years[0]) + '_Step=' + str(i-1) + ' to ' +str(i) + '.png'),
                            dpi=300, bbox_inches='tight', format='png')
                plt.close()
        else:
            time2 = np.max(np.argwhere(np.nancumsum(self.TakenMeasures['LCC']) < investmentlimit))
            for i in range(0,2):
                if i == 0: step1 = 0; step2 = time2; print(step1); print(step2)
                if i == 1: step1 = time2; step2 = -1; print(step1); print(step2)
                col = 0; mech = 0; line1 = {}; line2 = {}; mid = {}
                fig, ax = plt.subplots(figsize = figure_size)
                for j in mechanisms:
                    plotdata1 = copy.deepcopy(self.Probabilities[step1][years[0]].xs(j,level='mechanism',axis=0)).values
                    plotdata2 = copy.deepcopy(self.Probabilities[step2][years[0]].xs(j,level='mechanism',axis=0)).values
                    ydata1 = copy.deepcopy(plotdata1) #oude sterkte
                    ydata2 = copy.deepcopy(plotdata2) #nieuwe sterkte
                    for ij in range(0,len(plotdata1)):
                        ydata1 = np.insert(ydata1, ij*2, plotdata1[ij])
                        ydata2 = np.insert(ydata2, ij*2, plotdata2[ij])
                    line1[mech], = ax.plot(xticks1,ydata1, color=color[col], linestyle='-', alpha=alpha)
                    line2[mech], = ax.plot(xticks1,ydata2, color=color[col], linestyle='-', label=j)
                    mid[mech],   = ax.plot(middles,plotdata2, color = color[col], linestyle = '', marker = 'o')
                    col += 1
                    mech += 1
                for ik in cumlength: ax.axvline(x=ik, color='k', linestyle=':', alpha=0.5)
                ax.plot([0, max(cumlength)], [ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax']),
                                              ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax'])],
                                              'k--', label=label_target, linewidth=1)
                ax.legend(loc=1)
                ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                ax.set_xticklabels(labels_xticks)
                ax.tick_params(axis='x', rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis='both', bottom=False)
                ax.set_ylim([1.5, 8.5])
                ax.grid(axis='y')
                if flip: ax.invert_xaxis()
                if i == 0: plt.savefig(path.joinpath(str(2025+years[0]) + '_Begin to ' +str(int(investmentlimit)) +'.png'),
                                       dpi=300, bbox_inches='tight', format='png')
                if i == 1: plt.savefig(path.joinpath(str(2025+years[0]) + '_' + str(int(investmentlimit)) + ' to end.png'),
                            dpi=300, bbox_inches='tight', format='png')
                plt.close()




    def writeProbabilitiesCSV(self, path, type):
        # with open(path + '\\ReliabilityLog_' + type + '.csv', 'w') as f:
        for i in range(len(self.Probabilities)):
            name = path.joinpath('ReliabilityLog_' + type + '_Step' + str(i) + '.csv')
            # measurerow = self.TakenMeasures.iloc[i]['Section'] + ',' + self.TakenMeasures.iloc[i]['name'] + ',' + str(self.TakenMeasures.iloc[i]['params'])+ ',' + str(self.TakenMeasures.iloc[i]['LCC'])
            # f.write(measurerow)

            # self.TakenMeasures.iloc[i].to_csv(f, header=True)
            self.Probabilities[i].to_csv(path_or_buf=name, header=True)
    def determineRiskCostCurve(self,TrajectObject,PATH=False):
        if PATH:
            PATH.mkdir(parents=True, exist_ok=True)

        else:
            PATH = False
        if not hasattr(self, 'TakenMeasures'):
            raise TypeError('TakenMeasures not found')
        TR = []
        LCC = []
        if (self.type == 'Greedy') or (self.type == 'TC'): #do a loop

            LCC = np.cumsum(self.TakenMeasures['LCC'].values)
            count = 0
            for i in self.Probabilities:
                if PATH:
                    TR.append(calcLifeCycleRisks(i, self.r, np.max(TrajectObject.GeneralInfo['T']),
                                                 TrajectObject.GeneralInfo['FloodDamage'], dumpPt=PATH.joinpath('Greedy_step_' + str(count) + '.csv')))
                else:
                    TR.append(calcLifeCycleRisks(i, self.r, np.max(TrajectObject.GeneralInfo['T']),
                                                 TrajectObject.GeneralInfo['FloodDamage']))
                count += 1

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
        return TR, LCC

class GreedyStrategy(Strategy):
    def evaluate(self, traject, solutions, splitparams = False,setting='fast',BCstop=1,max_count = 150, f_cautious = 2):
        '''This is a faster version (supposedly). First we rebuild the old version. Afther that we add a different
        routine for properly dealing with overflow'''
        self.make_optimization_input(traject,solutions)
        #set start values:
        self.Cint_g[:,0] = 1
        self.Cint_h[:,0] = 1


        init_probability = {}
        init_overflow_risk = np.empty((self.opt_parameters['N'],self.opt_parameters['T']))
        init_geotechnical_risk = np.empty((self.opt_parameters['N'],self.opt_parameters['T']))
        for m in traject.GeneralInfo['Mechanisms']:
            init_probability[m] = np.empty((self.opt_parameters['N'], self.opt_parameters['T']))
            for n in range(0, self.opt_parameters['N']):
                init_probability[m][n, :] = self.Pf[m][n,0, :]
                if m == 'Overflow':
                    init_overflow_risk[n,:] = self.RiskOverflow[n,0,:]
                else:
                    init_geotechnical_risk[n,:] = self.RiskGeotechnical[n,0,:]

        # init_overflow_risk = np.sum(init_overflow_risk,axis=0)  #only sum for overflow as the max is of importance
        count = 0
        measure_list = []
        Probabilities = []
        risk_per_step = []
        cost_per_step = []
        cost_per_step.append(0)
        #TODo add existing investments
        SpentMoney = np.zeros([self.opt_parameters['N']])
        InitialCostMatrix = copy.deepcopy(self.LCCOption)
        BC_list = []
        while count < max_count:
            Probabilities.append(copy.deepcopy(init_probability))
            # start = time.time()
            init_risk = np.sum(np.max(init_overflow_risk,axis=0)) + np.sum(init_geotechnical_risk)
            risk_per_step.append(init_risk)
            # plt.plot(np.max(init_overflow_risk,axis=0),'b')
            # plt.plot(np.sum(init_geotechnical_risk,axis=0),'b--')
            # plt.ylim([0,400000])
            # plt.savefig('Risk at step ' + str(count) + '.png')
            # plt.close()
            # plt.plot(np.max(init_probability['Overflow'],axis=0),label=str(count))
            #first we compute the BC-ratio for each combination of Sh, Sg, for each section
            LifeCycleCost = np.full([self.opt_parameters['N'],self.opt_parameters['Sh'],self.opt_parameters['Sg']],1e99)
            TotalRisk = np.full([self.opt_parameters['N'],self.opt_parameters['Sh'],self.opt_parameters['Sg']],
                                init_risk)
            for n in range(0,self.opt_parameters['N']):
                #for each section, start from index 1 to prevent putting inf in top left cell
                for sg in range(1,self.opt_parameters['Sg']):
                    for sh in range(1, self.opt_parameters['Sh']):
                        if self.LCCOption[n,sh,sg] < 1e20:
                            LifeCycleCost[n,sh,sg] = copy.deepcopy(np.subtract(self.LCCOption[n,sh,sg],SpentMoney[n]))
                            new_overflow_risk,new_geotechnical_risk = evaluateRisk(traject.GeneralInfo['Mechanisms'],
                                                                                   copy.deepcopy(init_overflow_risk),
                                                                                   copy.deepcopy(
                                                                                       init_geotechnical_risk),self,n,sh,sg)
                            TotalRisk[n,sh,sg] = copy.deepcopy(np.sum(np.max(new_overflow_risk,axis=0)) +np.sum(
                                new_geotechnical_risk))
                        else:
                            pass
            # do not go back:
            LifeCycleCost = np.where(LifeCycleCost<0,1e99,LifeCycleCost)
            dR = np.subtract(init_risk,TotalRisk)
            BC = np.divide(dR, LifeCycleCost) #risk reduction/cost [n,sh,sg]
            TC = np.add(LifeCycleCost,TotalRisk)
            print(np.max(BC))
            if np.max(BC) > BCstop:
                #find the best combination
                Index_Best = np.unravel_index(np.argmax(BC),BC.shape)

                if setting == 'robust':
                    measure_list.append(Index_Best)
                    #update init_probability
                    init_probability = updateProbability(init_probability,self,Index_Best)

                elif (setting == 'fast') or (setting == 'cautious'):
                    BC_sections = np.empty((self.opt_parameters['N']))
                    #find best measure for each section
                    for n in range(0,self.opt_parameters['N']):
                        BC_sections[n] = np.max(BC[n,:,:])
                    BC_second = -np.partition(-BC_sections,2)[1]
                    if setting == 'fast':
                        indices = np.argwhere(BC[Index_Best[0]] - np.max([BC_second,1]) > 0)
                    elif setting == 'cautious':
                        indices = np.argwhere(np.divide(BC[Index_Best[0]],np.max([BC_second,1])) > f_cautious)
                    #a bit more cautious
                    if indices.shape[0]>1:
                        #take the investment that has the lowest total cost:


                        fast_measure = indices[np.argmin(TC[Index_Best[0]][(indices[:, 0], indices[:, 1])])]
                        Index_Best = (Index_Best[0], fast_measure[0], fast_measure[1])
                        measure_list.append(Index_Best)
                    else:
                        measure_list.append(Index_Best)
                BC_list.append(BC[Index_Best])
                init_probability = updateProbability(init_probability,self,Index_Best)
                # plt.plot(init_geotechnical_risk[Index_Best[0],:],'r')
                init_geotechnical_risk[Index_Best[0],:] = copy.deepcopy(self.RiskGeotechnical[Index_Best[0],
                                                          Index_Best[2],:])
                # plt.plot(init_geotechnical_risk[Index_Best[0], :],'b')
                # plt.savefig('Risk Geotechnical ' + str(Index_Best) + '.png')
                # plt.close()
                # plt.plot(init_overflow_risk[Index_Best[0],:],'r')
                init_overflow_risk[Index_Best[0],:] = copy.deepcopy(self.RiskOverflow[Index_Best[0],Index_Best[
                                                                                                        1],:])
                # plt.plot(init_overflow_risk[Index_Best[0],:],'b')
                # plt.savefig('Risk Overflow ' + str(Index_Best) + '.png')
                # plt.close()
                #TODO update risks
                SpentMoney[Index_Best[0]] += copy.deepcopy(LifeCycleCost[Index_Best])
                self.LCCOption[Index_Best] = 1e99

            else: #stop the search
                break
            count += 1
            if count == max_count:
                Probabilities.append(copy.deepcopy(init_probability))

            # print(count)
            # print(time.time()-start)
        self.LCCOption = copy.deepcopy(InitialCostMatrix)
        self.writeGreedyResults(traject,solutions,measure_list,BC_list,Probabilities)

    def writeGreedyResults(self,traject,solutions,measure_list,BC,Probabilities):
        TakenMeasuresHeaders = ['Section','option_index','LCC','BC','ID','name','yes/no','dcrest','dberm']
        sections = []
        LCC = []
        LCC2 = []
        LCC_invested = np.zeros((len(traject.Sections)))
        ID = []
        dcrest = []
        dberm = []
        yes_no = []
        option_index = []
        names = []
        #write the first line:
        sections.append(''); LCC.append(0); ID.append(''); dcrest.append('');
        dberm.append(''); yes_no.append(''); option_index.append(''); names.append('')
        BC.insert(0,0)
        for i in measure_list:
            sections.append(traject.Sections[i[0]].name)
            LCC.append(np.subtract(self.LCCOption[i],LCC_invested[i[0]])) #add costs and subtract the money already
            LCC2.append(self.LCCOption[i]) #add costs
            # spent
            LCC_invested[i[0]] += np.subtract(self.LCCOption[i],LCC_invested[i[0]])

            #get the ids
            ID1 = self.options_geotechnical[traject.Sections[i[0]].name].iloc[i[2] - 1]['ID'].values[0]
            ID2 = self.options_height[traject.Sections[i[0]].name].iloc[i[1] - 1]['ID'].values[0]
            if ID1[-1] == ID2:
                if (self.options_height[traject.Sections[i[0]].name].iloc[i[1] - 1]['dcrest'].values[0] == 0.0) and (
                        self.options_geotechnical[traject.Sections[i[0]].name].iloc[i[2] - 1]['dberm'].values[0] == 0.0):
                    ID.append(ID1[0])
                else:
                    ID.append(ID1)
            else:
                ValueError('warning, conflicting IDs found for measures')

            #get the parameters
            dcrest.append(self.options_height[traject.Sections[i[0]].name].iloc[i[1]-1]['dcrest'].values[0])
            dberm.append(self.options_geotechnical[traject.Sections[i[0]].name].iloc[i[2]-1]['dberm'].values[0])
            yes_no.append(self.options_geotechnical[traject.Sections[i[0]].name].iloc[i[2]-1]['yes/no'].values[0])

            #get the option_index
            option_df = self.options[traject.Sections[i[0]].name].loc[
                self.options[traject.Sections[i[0]].name]['ID'] == ID[-1]]
            if len(option_df) > 1:
                option_index.append(self.options[traject.Sections[i[0]].name].loc[
                    self.options[traject.Sections[i[0]].name]['ID'] == ID[-1]].loc[
                    self.options[traject.Sections[i[0]].name]['dcrest'] == dcrest[-1]].loc[
                    self.options[traject.Sections[i[0]].name]['dberm'] == dberm[-1]].loc[
                    self.options[traject.Sections[i[0]].name]['yes/no'] == yes_no[-1]].index.values[0])
            else:   #partial measure with no parameter variations
                option_index.append(self.options[traject.Sections[i[0]].name].loc[
                self.options[traject.Sections[i[0]].name]['ID'] == ID[-1]].index.values[0])
            #get the name
            names.append(solutions[traject.Sections[i[0]].name].MeasureTable.loc[solutions[traject.Sections[i[
                0]].name].MeasureTable['ID']==ID[-1]]['Name'].values[0][0])
        self.TakenMeasures = pd.DataFrame(list(zip(sections,option_index,LCC,BC,ID,names,yes_no,dcrest,dberm)),
                                          columns=TakenMeasuresHeaders)

        #writing the probabilities to self.Probabilities
        tgrid = copy.deepcopy(traject.GeneralInfo['T'])
        #make sure it doesnt exceed the data:
        tgrid[-1] = np.size(Probabilities[0]['Overflow'],axis=1)-1
        probabilities_columns = ['name','mechanism']+tgrid
        count = 0
        self.Probabilities = []
        for i in Probabilities:
            name = []
            mech = []
            probs = []
            for n in range(0,self.opt_parameters['N']):
                for m in traject.GeneralInfo['Mechanisms']:
                    name.append(traject.Sections[n].name)
                    mech.append(m)
                    probs.append(i[m][n,np.array(tgrid)])
                    pass
                name.append(traject.Sections[n].name)
                mech.append('Section')
                probs.append(np.sum(probs[-3:],axis=0))
            betas = np.array(ProbabilisticFunctions.pf_to_beta(probs))
            leftpart = pd.DataFrame(list(zip(name,mech)),columns = probabilities_columns[0:2])
            rightpart = pd.DataFrame(betas,columns=tgrid)
            combined = pd.concat((leftpart, rightpart), axis=1)
            combined = combined.set_index(['name','mechanism'])
            self.Probabilities.append(combined)

    def evaluate_backup(self, traject, solutions, splitparams = False,setting='fast',BCstop=0.1):
        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        #Step 2: calculate costs and risk reduction for each option
        #make a very basic traject dataframe of the current state that consists of current betas for each section per year
        BaseTrajectProbability = makeTrajectDF(traject, cols)
        count = 0
        measure_cols = ['Section', 'option_index', 'LCC', 'BC']
        if splitparams:
            TakenMeasures = pd.DataFrame(data=[[None, None, 0, None,None , None, None,None,None]],
                                         columns=measure_cols + ['ID','name','yes/no','dcrest','dberm'])
        else:
            TakenMeasures = pd.DataFrame(data=[[None,None, None, 0, None, None, None]],
                                         columns=measure_cols + ['ID','name', 'params'])

        #Calculate Total Cost for all options for all sections (this does not change based on values later
        # calculated so only has to be done once
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
            start = time.time()
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
            if not any(SectionMeasures['BC'] > BCstop):
                break

            #The following is faster but less robust:
            if setting == 'fast':
                id_opt = SectionMeasures.loc[SectionMeasures['BC'] > maxBC_others]['TotalCost'].idxmin()

            #This would be an in between:
            # if len(SectionMeasures.loc[SectionMeasures['BC'] > 3*maxBC_others])>2:
            #     id_opt = SectionMeasures.loc[SectionMeasures['BC'] > 3*maxBC_others]['TotalCost'].idxmin()
            # else:
            #     id_opt = SectionMeasures['BC'].idxmax()

            #This is most robust:
            if setting == 'robust':
                id_opt = SectionMeasures['BC'].idxmax()

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
            limit = ProbabilisticFunctions.beta_to_pf(np.min(sectionlevel.values.astype('float'), axis=0)) / p_factor
            indices = np.all(ProbabilisticFunctions.beta_to_pf(sectionlevel.values.astype('float')) > limit, axis=1)
            keys = []
            for i in range(0, len(indices)):
                if indices[i]:
                    keys.append(sections[i])
            print(time.time()-start)
        print('Run finished')
        self.Probabilities = Probability_steps
        self.TakenMeasures = TakenMeasures

class MixedIntegerStrategy(Strategy):
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
        TakenMeasures = TakenMeasures.sort_values('Section')
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

class ParetoFrontier(Strategy):
    def evaluate(self, traject, solutions, splitparams = False):
        #This is the core code of the optimization. This piece should probably be split into the different methods available.

        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)
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

class TargetReliabilityStrategy(Strategy):
    pass

    def evaluate(self, traject, solutions, OI_horizon=50, OI_year=0, splitparams = False):
        cols = list(solutions[list(solutions.keys())[0]].MeasureData['Section'].columns.values)

        # compute cross sectional requirements
        N_piping = 1 + (traject.GeneralInfo['aPiping'] * traject.GeneralInfo['TrajectLength'] / traject.GeneralInfo['bPiping'])
        N_stab = 1 + (traject.GeneralInfo['aStabilityInner'] * traject.GeneralInfo['TrajectLength'] / traject.GeneralInfo[
            'bStabilityInner'])
        N_overflow = 1
        beta_cs_piping = ProbabilisticFunctions.pf_to_beta(traject.GeneralInfo['Pmax'] * traject.GeneralInfo['omegaPiping'] / N_piping)
        beta_cs_stabinner = ProbabilisticFunctions.pf_to_beta(traject.GeneralInfo['Pmax'] * traject.GeneralInfo['omegaStabilityInner'] / N_stab)
        beta_cs_overflow = ProbabilisticFunctions.pf_to_beta(traject.GeneralInfo['Pmax'] * traject.GeneralInfo['omegaOverflow'] / N_overflow)

        # Rank sections based on 2075 Section probability
        beta_horizon = []

        for i in traject.Sections:
            beta_horizon.append(i.Reliability.SectionReliability.loc['Section'][str(OI_horizon)])

        section_indices = np.argsort(beta_horizon)

        measure_cols = ['Section', 'option_index', 'LCC', 'BC']

        if splitparams:
            TakenMeasures = pd.DataFrame(data=[[None, None, 0, None,None , None, None,None,None]],
                                         columns=measure_cols + ['ID','name','yes/no','dcrest','dberm'])
        else:
            TakenMeasures = pd.DataFrame(data=[[None,None, None, 0, None, None, None]],
                                         columns=measure_cols + ['ID','name', 'params'])
        # columns (section name and index in self.options[section])
        BaseTrajectProbability = makeTrajectDF(traject, cols)
        Probability_steps = [copy.deepcopy(BaseTrajectProbability)]
        TrajectProbability = copy.deepcopy(BaseTrajectProbability)

        for j in section_indices:
            i = traject.Sections[j]
            # convert beta_cs to beta_section in order to correctly search self.options[section] THIS IS CURRENTLY
            # INCONSISTENT WITH THE WAY IT IS CALCULATED
            beta_T_overflow = beta_cs_overflow
            beta_T_piping = ProbabilisticFunctions.pf_to_beta(ProbabilisticFunctions.beta_to_pf(beta_cs_piping) * (i.Length / traject.GeneralInfo['bPiping']))
            beta_T_stabinner = ProbabilisticFunctions.pf_to_beta(ProbabilisticFunctions.beta_to_pf(beta_cs_stabinner) * (i.Length / traject.GeneralInfo['bStabilityInner']))

            # find cheapest design that satisfies betatcs in 50 years from OI_year if OI_year is an int that is not 0
            if isinstance(OI_year, int):
                targetyear = 50  # OI_year + 50
            else:
                targetyear = 50

            # filter for overflow
            PossibleMeasures = copy.deepcopy(
                self.options[i.name].loc[self.options[i.name][('Overflow', targetyear)] > beta_T_overflow])

            # filter for piping
            PossibleMeasures = PossibleMeasures.loc[self.options[i.name][('Piping', targetyear)] > beta_T_piping]

            # filter for stabilityinner
            PossibleMeasures = PossibleMeasures.loc[PossibleMeasures[('StabilityInner', targetyear)] > beta_T_stabinner]

            # calculate LCC
            LCC = calcTC(PossibleMeasures, r=self.r, horizon=self.options[i.name]['Overflow'].columns[-1])

            # select measure with lowest cost
            idx = np.argmin(LCC)
            measure = PossibleMeasures.iloc[idx]

            # calculate achieved risk reduction & BC ratio compared to base situation
            R_base, dR, TR = calcTR(i.name, measure, TrajectProbability,
                                    original_section=TrajectProbability.loc[i.name], r=self.r, horizon=cols[-1],
                                    damage=traject.GeneralInfo['FloodDamage'])
            BC = dR / LCC[idx]

            if splitparams:
                name = IDtoName(measure['ID'].values[0],solutions[i.name].MeasureTable)
                data_opt = pd.DataFrame([[i.name, idx, LCC[idx], BC, measure['ID'].values[0], name,
                                          measure['yes/no'].values[
                    0], measure['dcrest'].values[0], measure['dberm'].values[0]]],columns=TakenMeasures.columns)
            else:
                data_opt = pd.DataFrame([[i.name, idx, LCC[idx], BC, measure['ID'].values[0], measure['name'].values[0],
                                          measure['params'].values[0]]],
                                        columns=TakenMeasures.columns)  # here we evaluate and pick the option that has the
                # lowest total cost and a BC ratio that is lower than any measure at any other section
            # if splitparams:
            #     data_opt = pd.DataFrame([[indd, id_opt, LCCdiff,
            #                               BC[indd][id_opt], self.options[indd].iloc[id_opt]['ID'].values[0],
            #                               IDtoName(self.options[indd].iloc[id_opt][
            #                                                              'ID'].values[0],solutions[
            #                                   indd].MeasureTable),
            #                               self.options[indd].iloc[id_opt]['yes/no'].values[0],self.options[indd].iloc[id_opt]['dcrest'].values[0],self.options[indd].iloc[id_opt]['dberm'].values[0]]],
            #                             columns=TakenMeasures.columns)
            # else:
            #     data_opt = pd.DataFrame([[indd, id_opt, LCCdiff, BC[indd][id_opt], self.options[indd].iloc[
            #         id_opt]['ID'].values[0], IDtoName(self.options[indd].iloc[id_opt]['ID'].values[0],solutions[
            #         indd].MeasureTable), self.options[indd].iloc[id_opt]['params'].values[0]]],
            #                             columns=TakenMeasures.columns)
            # Add to TakenMeasures
            TakenMeasures = TakenMeasures.append(data_opt)
            # Calculate new probabilities
            TrajectProbability = ImplementOption(i.name, TrajectProbability, measure)
            Probability_steps.append(copy.deepcopy(TrajectProbability))
        self.TakenMeasures = TakenMeasures
        self.Probabilities = Probability_steps
        pass
class SmartTargetReliabilityStrategy(Strategy):
    def evaluate(self, traject, solutions, OI_horizon=50, OI_year=0, splitparams = False):
        # TODO add a smarter OI version where the failure probability budget is partially redistributed of the mechanisms.

        # find section where it is most attractive to make 1 or multiple mechanisms to meet the cross sectional
        # reliability index
        # choice 1: geotechnical mechanisms ok for 2075

        # choice 2:also height ok for 2075
        print('SmartTargetReliability is not implemented yet')
        pass
