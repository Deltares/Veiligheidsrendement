## This script can calculate life-cycle reliability and costs for all measures for various mechanisms

#Import a bunch of packages
import matplotlib.pyplot as plt
import pandas as pd
import time
from DikeTraject import DikeTraject
from HelperFunctions import getMeasureTable
from RunModel import runFullModel
from pathlib import Path
from StrategyEvaluation import calcTrajectProb
from Solutions import Solutions
import cProfile
import shelve

def main():
    ## GENERAL SETTINGS
    timing = 1
    traject = '16-3 en 16-4'
    save_beta_measure_plots = False
    RunComputation = True
    years0 = [0, 19, 20, 50, 75, 100]
    mechanisms = ['Overflow', 'StabilityInner', 'Piping']
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland '
                r'SAFE\WJKlerk\SAFE\data\SAFE_totaal')
    # path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland '
    #             r'SAFE\WJKlerk\SAFE\data\Dijkwerkersdag')
    language = 'NL'

    if timing == 1:
        start = time.time()
    if timing == 1:
        start0 = time.time()

    ## MAKE TRAJECT OBJECT
    TestCase = DikeTraject('TestCase', traject)

    ## Run the model
    casename = 'nieuw_cautious_f=2_geenLE'
    directory = path.joinpath('Case_' + casename)

    ## READ ALL DATA
    ##First we read all the input data for the different sections. We store these in a Traject object.
    #Initialize a list of all sections that are of relevance (these start with DV).
    print('Start creating all the files and folders')
    TestCase.ReadAllTrajectInput(path, directory, years0,traject=traject, startyear=2025)


    #If you want to use intermediate data (from after step 2) you can uncomment the following snippet of code (and input it to runFullModel:
#This could be programmed more neatly of course...
    if RunComputation:
        AllStrategies, AllSolutions = runFullModel(TestCase, casename, path, directory, years=years0, timing=timing,
                                                   save_beta_measure_plots=save_beta_measure_plots,
                                                   LCRplot=False, language='NL',
                                                   types=['TC', 'OI'], OI_year=0) #,
        # TestCaseSolutions=TestCaseSolutions)

    #Same here: if you want to make plots based on existing results, uncomment the part underneath:
    #Open shelf
    if not RunComputation:
        filename = directory.joinpath('AfterStep1.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            TestCase = my_shelf[key]
        my_shelf.close()

        filename = directory.joinpath('AfterStep2.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            AllSolutions = my_shelf[key]
        my_shelf.close()

        filename = directory.joinpath('FINALRESULT.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            AllStrategies = my_shelf[key]
        my_shelf.close()

    #MAKING PLOTS:
    MeasureTable = getMeasureTable(AllSolutions)
    # TestCase.setProbabilities()
    #Plot the beta-t:
    beta_t = []
    step = 0

    #plot beta time for all measure steps for each strategy
    setting = ['beta', 'prob']       #PLOT FOR BETA AND OR PROBABILITY

    #plot beta costs for t=0
    figure_size = (12, 7)

    for i in setting:
        # LCC-beta for t = 0
        plt.figure(101, figsize=figure_size)
        AllStrategies[0].plotBetaCosts(TestCase, path=directory.joinpath('figures'), typ='multi', fig_id=101, symbolmode='on', linecolor='b', labels='TC', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)
        AllStrategies[1].plotBetaCosts(TestCase, path=directory.joinpath('figures'), typ='multi', fig_id=101, last='yes', symbolmode='on', labels='OI', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)

        # LCC-beta for t=50
        plt.figure(102, figsize=figure_size)
        AllStrategies[0].plotBetaCosts(TestCase, t=50, path=directory.joinpath('figures'), typ='multi', fig_id=102, symbolmode='on', linecolor='b', labels='TC', MeasureTable=MeasureTable, last='yes', beta_or_prob=i, outputcsv=True)
        AllStrategies[1].plotBetaCosts(TestCase, t=50, path=directory.joinpath('figures'), typ='multi', fig_id=102, symbolmode='on', labels='OI', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)

        # Costs2025-beta
        plt.figure(103, figsize=figure_size)
        AllStrategies[0].plotBetaCosts(TestCase, cost_type='Initial', path=directory.joinpath('figures'),
                                       typ='multi', fig_id=103, symbolmode='on', linecolor='b', labels='TC',
                                       MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)
        AllStrategies[1].plotBetaCosts(TestCase, cost_type='Initial', path=directory.joinpath('figures'),
                                       typ='multi', fig_id=103, last='yes', symbolmode='on', labels='OI', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)

    # AllStrategies[0].plotInvestmentSteps(TestCase, path= directory.joinpath('figures'),figure_size = (12,6),years=[0],
    #                                      flip=True)
    AllStrategies[0].plotInvestmentSteps(TestCase, investmentlimit= 40e6, path= directory.joinpath('figures'),
                                         figure_size = (12,6),years=[0],flip=True)
    # TestCaseStrategyOI.plotInvestmentSteps(TestCase, path= pad + '\\Case_' + casename + '\\OI',figure_size = (6,4))

    ## write a LOG of all probabilities for all steps:
    AllStrategies[0].writeProbabilitiesCSV(path=directory.joinpath('results', 'investment_steps'), type='TC')
    AllStrategies[1].writeProbabilitiesCSV(path=directory.joinpath('results', 'investment_steps'), type='OI')
    ps = []

    for i in AllStrategies[1].Probabilities:
        beta_t, p_t = calcTrajectProb(i, horizon=100)
        ps.append(p_t)

    ps = pd.DataFrame(ps, columns=range(100))
    ps.to_csv(path_or_buf=directory.joinpath('results', 'investment_steps', 'PfT_OI.csv'))
    ps = []

    for i in AllStrategies[0].Probabilities:
        beta_t, p_t = calcTrajectProb(i, horizon=100)
        ps.append(p_t)

    ps = pd.DataFrame(ps, columns=range(100))
    ps.to_csv(path_or_buf=directory.joinpath('results', 'investment_steps', 'PfT_TC.csv'))

    if timing == 1:
        end = time.time()

    if timing == 1:
        print("time elapsed: " + str(end - start) + ' seconds')

    if timing == 1:
        start = time.time()

    if timing == 1:
        end = time.time()

    if timing == 1:
        print("Overall time elapsed: " + str(end - start0) + ' seconds')

if __name__ == '__main__':
    main()
    # cProfile.run('main()','InvestmentsSAFE.profile')