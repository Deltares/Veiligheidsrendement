#import modules:
from Strategy import ImplementOption, GreedyStrategy, TargetReliabilityStrategy
from Solutions import Solutions
import time
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import shelve
import copy
import numpy as np
import cProfile
import os
from scipy.stats import norm
from HelperFunctions import replaceNames
import ProbabilisticFunctions
import pandas as pd

# this is sort of the main script for any calculation for SAFE. It contains all the required steps:
def runFullModel(TestCase, run_number, path, directory, mechanisms=['Overflow', 'StabilityInner', 'Piping'],
                 years=[0, 1, 10, 20, 40, 50], timing=0, save_beta_measure_plots=False, LCRplot = False, shelves=1,
                 types=['TC', 'SmartOI', 'OI'], language='NL', TestCaseSolutions=False, t_start=2025, OI_year=0):
    if timing == 1:
        start = time.time()

    #Make a few dirs if they dont exist yet:
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
        directory.joinpath('figures').mkdir(parents=True, exist_ok=True)
        directory.joinpath('results').mkdir(parents=True, exist_ok=True)

    ## STEP 1: SAFETY ASSESSMENT
    print('Start step 1: safety assessment')

    for i in range(len(TestCase.Sections)):
        TestCase.Sections[i].Reliability.Load.NormWaterLevel = ProbabilisticFunctions.getDesignWaterLevel(
            TestCase.Sections[i].Reliability.Load,TestCase.GeneralInfo['Pmax'])
        for j in mechanisms:
            TestCase.Sections[i].Reliability.Mechanisms[j].generateLCRProfile(
                TestCase.Sections[i].Reliability.Load, mechanism=j, trajectinfo=TestCase.GeneralInfo)
        TestCase.Sections[i].Reliability.calcSectionReliability(TrajectInfo=TestCase.GeneralInfo,
                                                                       length=TestCase.Sections[i].Length)
        if LCRplot:
            #Plot the initial reliability-time:
            plt.figure(1)
            [TestCase.Sections[i].Reliability.Mechanisms[j].drawLCR(label=j, type='Standard', tstart=t_start) for j in
             mechanisms]
            plt.plot([t_start, t_start + np.max(years)],
                     [ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax']), ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax'])],
                     'k--', label='Norm')
            plt.legend()
            plt.title(TestCase.Sections[i].name)
            if not directory.joinpath('figures', TestCase.Sections[i].name).is_dir():
                directory.joinpath('figures', TestCase.Sections[i].name).mkdir(parents=True, exist_ok=True)
                directory.joinpath('figures', TestCase.Sections[i].name, 'Initial')

            plt.savefig(directory.joinpath('figures', TestCase.Sections[i].name, 'Initial', 'InitialSituation' + '.png'), bbox_inches='tight')
            plt.close()

    TestCase.setProbabilities()
    # plot reliability and failure probability for entire traject:
    figsize = (12, 4)
    TestCase.plotAssessment(PATH=directory, fig_size=figsize, language=language, flip='off',
                            draw_targetbeta='off', beta_or_prob='beta', outputcsv=True,
                            years = [0, 20, 50],
                            last=True)
    # TestCase.plotReliabilityofDikeTraject(PATH=directory, fig_size=figsize, language=language, flip='off', draw_targetbeta='off', beta_or_prob='prob', first=False, last=True)

    print('Finished step 1: assessment of current situation')

    if timing == 1:
        end = time.time()

    if timing == 1:
        print("Time elapsed: " + str(end - start) + ' seconds')

    if timing == 1:
        start = time.time()

    #store stuff:
    if shelves == 1:
        # Save intermediate results to shelf:
        filename = directory.joinpath('AfterStep1.out')
        # make shelf
        my_shelf = shelve.open(str(filename), 'n')
        my_shelf['TestCase'] = locals()['TestCase']
        my_shelf.close()

        # open shelf
        # my_shelf = shelve.open(filename)
        # for key in my_shelf:
        #     locals()[key]=my_shelf[key]
        # my_shelf.close()

    ## STEP 2: INITIALIZE AND EVALUATE MEASURES FOR EACH SECTION
    # Result: Measures object with Section name and beta-t-euro relations for each measure
    if not TestCaseSolutions:
        AllSolutions = {}

        # Calculate for each measure the cost-reliability-time relations
        for i in TestCase.Sections:
            AllSolutions[i.name] = Solutions(i)
            AllSolutions[i.name].fillSolutions(path.joinpath(i.name + '.xlsx'))
            AllSolutions[i.name].evaluateSolutions(i, TestCase.GeneralInfo, geometry_plot=False, trange=years, plot_dir=directory.joinpath('figures', i.name))
            #NB: geometry_plot = True plots the soil reinforcement geometry, but costs a lot of time!
    else:
        AllSolutions = TestCaseSolutions
    print('Finished step 2: evaluation of measures')
    if timing == 1:
        end = time.time()

    if timing == 1:
        print("Time elapsed: " + str(end - start) + ' seconds')

    if timing == 1:
        start = time.time()

    for i in TestCase.Sections:
        AllSolutions[i.name].SolutionstoDataFrame(filtering='off',splitparams=True)

    #possibly plot beta(t)-cost for all measures at a section:
    if save_beta_measure_plots:
        betaind_array = []

        for i in years:
            betaind_array.append('beta' + str(i))

        plt_mech = ['Section', 'Piping', 'StabilityInner', 'Overflow']

        for i in TestCase.Sections:
            for betaind in betaind_array:
                for mech in plt_mech:
                    requiredbeta = ProbabilisticFunctions.pf_to_beta(TestCase.GeneralInfo['Pmax'] * (i.Length / TestCase.GeneralInfo['TrajectLength']))
                    plt.figure(1001)
                    AllSolutions[i.name].plotBetaTimeEuro(mechanism=mech, beta_ind=betaind, sectionname=i.name, beta_req=requiredbeta)
                    plt.savefig(directory.joinpath('figures', i.name, 'Measures', mech + '_' + betaind + '.png'), bbox_inches='tight')
                    plt.close(1001)

        print('Finished making beta plots')

    if timing == 1:
        end = time.time()

    if timing == 1:
        print("Time elapsed: " + str(end - start) + ' seconds')

    if timing == 1:
        start = time.time()



    ## STEP 3: EVALUATE THE STRATEGIES
    AllStrategies = []

    for i in types:
        if i == 'TC':
            # Initialize a strategy type (i.e combination of objective & constraints)
            TestCaseStrategyTC = GreedyStrategy('TC')
            # Combine available measures
            TestCaseStrategyTC.combine(TestCase, AllSolutions, filtering='off',splitparams=True)

            if shelves == 1:
                # Store intermediate results:
                filename = directory.joinpath('AfterStep2.out')
                #
                # make shelf
                my_shelf = shelve.open(str(filename), 'n')
                my_shelf['AllSolutions'] = locals()['AllSolutions']
                my_shelf.close()
            if timing == 1:
                end = time.time()
                print('Combine step for TC')
                print("Time elapsed: " + str(end - start) + ' seconds')

            # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
            # TestCaseStrategyTC.evaluate(TestCase, AllSolutions,splitparams=True,setting='fast')
            TestCaseStrategyTC.evaluate(TestCase, AllSolutions,splitparams=True,setting='cautious', f_cautious=2,
                                        max_count = 300)
            # TestCaseStrategyTC.evaluate_backup(TestCase, AllSolutions,splitparams=True,setting='robust')

            # plot beta time for all measure steps for each strategy
            TestCaseStrategyTC.plotBetaTime(TestCase, typ='single', path=directory, horizon=np.max(years))

            # plot beta costs for t=0
            plt.figure(101, figsize=(20, 10))
            TestCaseStrategyTC.plotBetaCosts(TestCase, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            plt.close(101)

            # plot beta costs for t=50
            plt.figure(102, figsize=(20, 10))
            TestCaseStrategyTC.plotBetaCosts(TestCase, t=50, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            TestCaseStrategyTC = replaceNames(TestCaseStrategyTC, AllSolutions)
            plt.close(102)

            # write to csv's
            for j in TestCaseStrategyTC.options:
                TestCaseStrategyTC.options[j].to_csv(directory.joinpath('results', j + '_Options_TC.csv'))
            [TR,LCC] = TestCaseStrategyTC.determineRiskCostCurve(TestCase)
            pd.DataFrame([TR,LCC]).to_csv(directory.joinpath('results','TotalRiskCost.csv'))
            TestCaseStrategyTC.TakenMeasures.to_csv(directory.joinpath('results', 'TakenMeasures_TC.csv'))
            TestCaseStrategyTC.makeFinalSolution(directory.joinpath('results', 'FinalMeasures_TC.csv'))
            AllStrategies.append(TestCaseStrategyTC)

        elif i == 'OI':
            # Initialize a strategy type (i.e combination of objective & constraints)
            TestCaseStrategyOI = TargetReliabilityStrategy('OI')
            # Combine available measures
            TestCaseStrategyOI.combine(TestCase, AllSolutions, filtering='off', OI_year=OI_year,splitparams=True)
            if timing == 1:
                end = time.time()

            if timing == 1:
                print('Combine step for OI')

            if timing == 1:
                print("Time elapsed: " + str(end - start) + ' seconds')

            # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
            TestCaseStrategyOI.evaluate(TestCase, AllSolutions, OI_year=OI_year,splitparams=True)

            # plot beta time for all measure steps for each strategy
            TestCaseStrategyOI.plotBetaTime(TestCase, typ='single', path=directory, horizon=np.max(years))

            # plot beta costs for t=0
            plt.figure(101, figsize=(20, 10))
            TestCaseStrategyOI.plotBetaCosts(TestCase, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            plt.close(101)

            # plot beta costs for t=50
            plt.figure(102, figsize=(20, 10))
            TestCaseStrategyOI.plotBetaCosts(TestCase, t=50, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            TestCaseStrategyOI = replaceNames(TestCaseStrategyOI, AllSolutions)
            plt.close(102)

            # write to csv's
            for j in TestCaseStrategyOI.options:
                TestCaseStrategyOI.options[j].to_csv(directory.joinpath('results', j + '_Options_OI.csv'))

            TestCaseStrategyOI.TakenMeasures.to_csv(directory.joinpath('results', 'TakenMeasures_OI.csv'))
            AllStrategies.append(TestCaseStrategyOI)

    if shelves == 1:
        # Store final results
        filename = directory.joinpath('FINALRESULT.out')

        # make shelf
        my_shelf = shelve.open(str(filename), 'n')
        my_shelf['TestCase'] = locals()['TestCase']
        my_shelf['AllSolutions'] = locals()['AllSolutions']
        my_shelf['AllStrategies'] = locals()['AllStrategies']

        my_shelf.close()

    return AllStrategies, AllSolutions


#Separate routine used for GeoRisk paper
def runNature(strategy, nature, traject, nature_solutions, directory=None, shelves=1):
    MeasureTable = getMeasureTable(nature_solutions)
    years = list(nature.Probabilities[0].columns)
    nature_orig = copy.deepcopy(nature)
    #Get the strategy from strategy and put it in nature (TakenMeasures) This is the strategy
    nature.TakenMeasures = strategy
    new_Probabilities = []
    BaseTrajectProbability = copy.deepcopy(nature_orig.Probabilities[0])
    #Evaluate for each step in strategy the total risk and cost
    for i in range(0,len(strategy)):
        if i == 0:
            new_Probabilities.append(BaseTrajectProbability)
            #write the base case
        else:
            BaseProbability = copy.deepcopy(new_Probabilities[-1])
            TrajectProbability = ImplementOption(strategy.ix[i]['Section'],
                                                 BaseProbability, nature.options[strategy.ix[i]['Section']].iloc[strategy.ix[i]['option_index']])
            new_Probabilities.append(copy.deepcopy(TrajectProbability))
            #potentially add changed BC values
    nature.Probabilities = new_Probabilities

    if not os.path.exists(directory):
        os.makedirs(directory + '\\figures')
        os.makedirs(directory + '\\results')

    plt.figure(101,figsize=(8,4))
    nature_orig.plotBetaCosts(traject,path = directory + '\\figures',
                              typ='multi',fig_id=101,linecolor='b',
                              name='NatureBeliefComparison',symbolmode='on',MeasureTable=MeasureTable,labels='With monitoring')           #The optimized strategy
    nature.plotBetaCosts(traject,path = directory + '\\figures' ,
                         typ='multi',last='yes',fig_id=101,linestyle='--',
                         name='NatureBeliefComparison',symbolmode='on',MeasureTable=MeasureTable,labels='Without monitoring')    #The strategy without knowing nature

    plt.close(101)
    #write to a separate folder\


    if shelves == 1:
        # Store final results
        filename = directory + '\\FINALRESULT.out'
        # make shelf
        my_shelf = shelve.open(filename, 'n')
        my_shelf['TestCase'] = locals()['traject']
        my_shelf['TestCaseStrategyTC'] = locals()['nature']
        my_shelf.close()
        # plot beta time for all measure steps for each strategy


    # plot beta costs for t=0
    plt.figure(101, figsize=(20, 10))
    nature.plotBetaCosts(traject, path=directory, typ='single', fig_id=101,
                         horizon=np.max(years),symbolmode='on',MeasureTable=MeasureTable)
    # plot beta costs for t=50
    plt.figure(102, figsize=(20, 10))
    nature.plotBetaCosts(traject, t=50, path=directory, typ='single', fig_id=101,
                         horizon=np.max(years),symbolmode='on',MeasureTable=MeasureTable)
    nature = replaceNames(nature,nature_solutions)
    #write to csv's
    for i in nature.options: nature.options[i].to_csv(directory + '\\results\\' + i + '_Options_TC.csv')
    nature.TakenMeasures.to_csv(directory + '\\results\\' + 'TakenMeasures_TC.csv')
    return nature
            #re-evaluate Probabilities
    #save the adapted nature with extension _base