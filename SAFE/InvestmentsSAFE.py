""" This script can calculate life-cycle reliability and costs for all measures for various mechanisms. It uses both a target reliability based approach and a greedy search optimization"""

import sys
sys.path.append('..')

#Import a bunch of packages
import matplotlib.pyplot as plt
import pandas as pd
import time
from DikeTraject import DikeTraject
from HelperFunctions import getMeasureTable
from RunModel import runFullModel
from pathlib import Path
from StrategyEvaluation import calcTrajectProb
import config
import shelve
from os import path

#General and global settings:
global timing, traject, path,casename, directory


def main():
    ## GENERAL SETTINGS
    RunComputation = True

    if config.timing:
        start = time.time()
        start_overall = time.time()

    ## MAKE TRAJECT OBJECT
    TrajectObject = DikeTraject()

    ## READ ALL DATA
    ##First we read all the input data for the different sections. We store these in a Traject object.
    #Initialize a list of all sections that are of relevance (these start with DV).
    print('Start creating all the files and folders')
    TrajectObject.ReadAllTrajectInput()

    if RunComputation:
        #compute everything
        AllStrategies, AllSolutions = runFullModel(TrajectObject)
    else:
        #load existing results
        filename = config.directory.joinpath('AfterStep1.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            TrajectObject = my_shelf[key]
        my_shelf.close()

        filename = config.directory.joinpath('AfterStep2.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            AllSolutions = my_shelf[key]
        my_shelf.close()

        filename = config.directory.joinpath('FINALRESULT.out')
        my_shelf = shelve.open(str(filename))
        for key in my_shelf:
            AllStrategies = my_shelf[key]
        my_shelf.close()

    #Now some general output figures and csv's are generated:

    #First make a table of all the solutions:
    MeasureTable = getMeasureTable(AllSolutions)

    #plot beta costs for t=0
    figure_size = (12, 7)

    # LCC-beta for t = 0
    plt.figure(101, figsize=figure_size)
    AllStrategies[0].plotBetaCosts(TrajectObject, fig_id=101, series_name=config.design_methods[0],MeasureTable=MeasureTable,color='b')
    AllStrategies[1].plotBetaCosts(TrajectObject, fig_id=101, series_name=config.design_methods[1],MeasureTable=MeasureTable,last='yes')
    plt.savefig(config.directory.joinpath('Priority order Beta vs LCC_' + str(config.t_0) + '.png'), dpi=300, bbox_inches='tight', format='png')

    # LCC-beta for t=50
    plt.figure(102, figsize=figure_size)
    AllStrategies[0].plotBetaCosts(TrajectObject, t=50, fig_id=102, series_name=config.design_methods[0], MeasureTable=MeasureTable, color='b' )
    AllStrategies[1].plotBetaCosts(TrajectObject, t=50, fig_id=102, series_name=config.design_methods[1], MeasureTable=MeasureTable, last='yes')
    plt.savefig(config.directory.joinpath('Priority order Beta vs LCC_' + str(config.t_0+50) + '.png'), dpi=300, bbox_inches='tight', format='png')

    # Costs2025-beta
    plt.figure(103, figsize=figure_size)
    AllStrategies[0].plotBetaCosts(TrajectObject, cost_type='Initial', fig_id=103, series_name=config.design_methods[0], MeasureTable=MeasureTable, color='b')
    AllStrategies[1].plotBetaCosts(TrajectObject, cost_type='Initial', fig_id=103, series_name=config.design_methods[1], MeasureTable=MeasureTable, last='yes')
    plt.savefig(config.directory.joinpath('Priority order Beta vs Costs_' + str(config.t_0+50) + '.png'), dpi=300, bbox_inches='tight', format='png')

    AllStrategies[0].plotInvestmentLimit(TrajectObject, investmentlimit= 20e6, path= config.directory.joinpath('figures'), figure_size = (12,6),years=[0],flip=True)

    ## write a LOG of all probabilities for all steps:
    AllStrategies[0].writeReliabilityToCSV(path=config.directory.joinpath('results', 'investment_steps'), type=config.design_methods[0])
    AllStrategies[1].writeReliabilityToCSV(path=config.directory.joinpath('results', 'investment_steps'), type=config.design_methods[1])

    for count, Strategy in enumerate(AllStrategies):
        ps = []
        for i in Strategy.Probabilities:
            beta_t, p_t = calcTrajectProb(i, horizon=100)
            ps.append(p_t)
        pd.DataFrame(ps, columns=range(100)).to_csv(path_or_buf=config.directory.joinpath('results', 'investment_steps', 'PfT_' + config.design_methods[count] + '.csv'))

    if config.timing:
        print("Time elapsed: " + str(time.time() - start) + ' seconds')
        print("Overall time elapsed: " + str(time.time() - start_overall) + ' seconds')

if __name__ == '__main__':
    main()
    # cProfile.run('main()','InvestmentsSAFE.profile')