## This script can calculate life-cycle reliability and costs for all measures for various mechanisms

#Import a bunch of packages
import matplotlib.pyplot as plt
import os
import pandas as pd
import shelve
import time
from DikeClasses import DikeTraject
from HelperFunctions import getMeasureTable
from RunModel import runFullModel
from pathlib import Path
from scipy.stats import norm
from StrategyEvaluation import calcTrajectProb

def main():
    ## GENERAL SETTINGS
    timing = 1
    traject = '16-4'
    save_beta_measure_plots = 0
    years0 = [0, 19, 20, 50, 75, 100]
    mechanisms = ['Overflow', 'StabilityInner', 'Piping']
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland '
                r'SAFE\WJKlerk\SAFE\data\Dijkwerkersdag')
    language = 'NL'

    if timing == 1:
        start = time.time()
    if timing == 1:
        start0 = time.time()

    ## MAKE TRAJECT OBJECT
    TestCase = DikeTraject('TestCase', traject)

    ## Run the model
    casename = 'SAFE'
    directory = path.joinpath('Case_' + casename)

    ## READ ALL DATA
    ##First we read all the input data for the different sections. We store these in a Traject object.
    #Initialize a list of all sections that are of relevance (these start with DV).
    print('Start creating all the files and folders')
    TestCase.ReadAllTrajectInput(path, directory, traject, years0, startyear=2025)

#If you want to use intermediate data (from after step 2) you can uncomment the following snippet of code (and input it to runFullModel:
#This could be programmed more neatly of course...

    # filename = directory.joinpath('AfterStep2.out')
    # my_shelf = shelve.open(str(filename))
    # for key in my_shelf:
    #     locals()[key] = my_shelf[key]
    # my_shelf.close()

    AllStrategies, AllSolutions = runFullModel(TestCase, casename, path, directory, years=years0, timing=timing, save_beta_measure_plots=save_beta_measure_plots, language='NL', types=['TC', 'OI'], OI_year=0) #,TestCaseSolutions=TestCaseSolutions)

    #Same here: if you want to make plots based on existing results, uncomment the part underneath:

    # # #Open shelf
    # filename = directory.joinpath('FINALRESULT.out')
    # my_shelf = shelve.open(str(filename))
    # for key in my_shelf:
    #     locals()[key] = my_shelf[key]
    # my_shelf.close()

    #MAKING PLOTS:
    MeasureTable = getMeasureTable(AllSolutions)

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
        AllStrategies[0].plotBetaCosts(TestCase, cost_type='Initial', path=directory.joinpath('figures'), typ='multi', fig_id=103, symbolmode='on', linecolor='b', labels='TC', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)
        AllStrategies[1].plotBetaCosts(TestCase, cost_type='Initial', path=directory.joinpath('figures'), typ='multi', fig_id=103, last='yes', symbolmode='on', labels='OI', MeasureTable=MeasureTable, beta_or_prob=i, outputcsv=True)

    #This piece makes sort of a 'movie' of all reinforcement steps:
    # if not 'TC' in os.listdir(pad + '\\Case_' + casename): os.makedirs(pad + '\\Case_' + casename + '\\TC')
    # if not 'OI' in os.listdir(pad + '\\Case_' + casename): os.makedirs(pad + '\\Case_' + casename + '\\OI')
    # TestCaseStrategyTC.plotInvestmentSteps(TestCase, path= pad + '\\Case_' + casename + '\\TC',figure_size = (6,4))
    # TestCaseStrategyOI.plotInvestmentSteps(TestCase, path= pad + '\\Case_' + casename + '\\OI',figure_size = (6,4))

    ## write a LOG of all probabilities for all steps:
    AllStrategies[0].writeProbabilitiesCSV(path=directory.joinpath('results', 'investment_steps'), type='TC')
    AllStrategies[1].writeProbabilitiesCSV(path=directory.joinpath('results', 'investment_steps'), type='OI')
    ps = []

    for i in AllStrategies[1].Probabilities:
        beta_t, p_t = calcTrajectProb(i, horizon=100)
        ps.append(p_t)

    ps = pd.DataFrame(ps, columns=range(101))
    ps.to_csv(path_or_buf=directory.joinpath('results', 'investment_steps', 'PfT_OI.csv'))
    ps = []

    for i in AllStrategies[0].Probabilities:
        beta_t, p_t = calcTrajectProb(i, horizon=100)
        ps.append(p_t)

    ps = pd.DataFrame(ps, columns=range(101))
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