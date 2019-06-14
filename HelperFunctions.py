try:
    import cPickle as pickle
except:
    import pickle
import shelve
import copy
import pandas as pd
from openturns.viewer import View
import matplotlib.pyplot as plt
import openturns as ot
from pandas import DataFrame
import numpy as np
import os
from scipy.stats import norm
import time
from StrategyEvaluation import Solutions, Strategy, ImplementOption

## This .py file contains a bunch of functions that are useful but do not fit under any of the other .py files.

# write to file with cPickle/pickle (as binary)
def ld_writeObject(filePath,object):
    f=open(filePath,'wb')
    newData = pickle.dumps(object, 1)
    f.write(newData)
    f.close()

#Used to read a pickle file which can objects
def ld_readObject(filePath):
    # This script can load the pickle file so you have a nice object (class or dictionary
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

#Helper function to flatten a nested dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def drawAlphaBarPlot(resultList,xlabels = None, Pvalues = None, method = 'MCS', suppress_ind = None, title = None):
    #draw stacked bars for the importance factors of a list of openturns FORM and results
    idx = np.nonzero(Pvalues)[0][0]
    if method == 'MCS':
        labels = resultList[idx].getImportanceFactors().getDescription()
    elif method == 'FORM':
        labels = resultList[idx].getImportanceFactors(ot.AnalyticalResult.CLASSICAL).getDescription()

    alphas = []
    for i in range(idx, len(resultList)):
        alpha = list(resultList[i].getImportanceFactors(ot.AnalyticalResult.CLASSICAL)) if method == 'FORM' else list(resultList[i].getImportanceFactors())
        if suppress_ind != None:
            for ix in suppress_ind:
                alpha[ix] = 0.0
        alpha = np.array(alpha)/np.sum(alpha)
        alphas.append(alpha)

    alfas = DataFrame(alphas, columns=labels)
    alfas.plot.bar(stacked=True, label=xlabels)
    # print('done')
    if Pvalues != None:
        plt.plot(range(0, len(xlabels)), Pvalues, 'b', label='Fragility Curve')

    xlabels = ['{:4.2f}'.format(xlabels[i]) for i in range(0, len(xlabels))]
    plt.xticks(range(0, len(xlabels)), xlabels)
    plt.legend()
    plt.title(title)
    plt.show()
    #TO BE DONE: COmpare a reference case to Matlab

def calc_r_exit(h_exit,k,d_cover,D,wl,Lbase,Lachter,Lfore = 0):
    # k = 0.0001736111
    # h_exit = 2.5
    # wl = 6.489
    # Lbase = 36
    # Lachter = 5.65
    lambda2 = np.sqrt(((k* 86400) * D) * (d_cover / 0.01))
    #slight modification: foreshore is counted as dijkzate
    phi2 = h_exit + (wl - h_exit)*((lambda2*np.tanh(2000/lambda2))/(lambda2*np.tanh(Lfore/lambda2)+Lbase+Lachter+lambda2*np.tanh(2000/lambda2)))
    r_exit = (phi2-h_exit)/(wl-h_exit)
    return r_exit

def adaptInput(grid_data,monitored_sections,BaseCase):
    CasesGeoRisk = []
    for i, row in grid_data.iterrows():
        CasesGeoRisk.append(copy.deepcopy(BaseCase))
        # adapt k-value
        for j in CasesGeoRisk[-1].Sections:
            if j.name in monitored_sections:
                for ij in list(j.Reliability.Mechanisms['Piping'].Reliability.keys()):
                    j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['k'] = row['k']
                    wl = \
                    np.array(j.Reliability.Load.distribution.computeQuantile(1 - CasesGeoRisk[-1].GeneralInfo['Pmax']))[
                        0]
                    new_r_exit = calc_r_exit(j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['h_exit'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['k'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['d_cover'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['D'],
                                             wl,
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['Lvoor'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['Lachter'])
                    j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['r_exit'] = new_r_exit

            CasesGeoRisk[-1].GeneralInfo['P_scen'] = row['p']                # print(str(row['k']) + '   ' + str(new_r_exit))
    return CasesGeoRisk

def replaceNames(TestCaseStrategy, TestCaseSolutions):
    TestCaseStrategy.TakenMeasures =TestCaseStrategy.TakenMeasures.reset_index(drop=True)
    for i in range(1, len(TestCaseStrategy.TakenMeasures)):
        # names = TestCaseStrategy.TakenMeasures.iloc[i]['name']
        #
        # #change: based on ID and get Names from new table.
        # if isinstance(names, list):
        #     for j in range(0, len(names)):
        #         names[j] = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names[j]].parameters['Name']
        # else:
        #     names = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names].parameters['Name']
        id = TestCaseStrategy.TakenMeasures.iloc[i]['ID']
        if isinstance(id,list): id = '+'.join(id)

        section = TestCaseStrategy.TakenMeasures.iloc[i]['Section']
        name = TestCaseSolutions[section].MeasureTable.loc[TestCaseSolutions[section].MeasureTable['ID'] == id]['Name'].values
        TestCaseStrategy.TakenMeasures.at[i, 'name'] = name
    return TestCaseStrategy


# this is sort of the main script for any calculation for SAFE. It contains all the required steps:
def runFullModel(TestCase, run_number, base_dir, mechanisms=['Overflow', 'StabilityInner', 'Piping'],
             years=[0, 1, 10, 20, 40, 50], timing=0, save_beta_measure_plots=0,shelves=1,
             types = ['TC', 'SmartOI', 'OI'],language='NL',TestCaseSolutions = None,t_start=2025,OI_year = 0):
    if timing == 1: start = time.time()

    #make a few dirs if they dont exist yet:
    if isinstance(run_number,str):
        directory = base_dir + '\\Case_' + run_number
    else:
        directory = base_dir + '\\Case_' + str(run_number)
    if not os.path.exists(directory):
        os.makedirs(directory + '\\figures')
        os.makedirs(directory + '\\results')

    ## STEP 1: SAFETY ASSESSMENT
    print('Start step 1: safety assessment')
    for i in range(0, len(TestCase.Sections)):
        for j in mechanisms:
            TestCase.Sections[i].Reliability.Mechanisms[j].generateLCRProfile(
                TestCase.Sections[i].Reliability.Load, mechanism=j, trajectinfo=TestCase.GeneralInfo)
        TestCase.Sections[i].Reliability.calcSectionReliability(TrajectInfo=TestCase.GeneralInfo,
                                                                       length=TestCase.Sections[i].Length)
        #Plot the initial reliability-time:
        plt.figure(1)
        [TestCase.Sections[i].Reliability.Mechanisms[j].drawLCR(label=j, type='Standard', tstart=t_start) for j in
         mechanisms]
        plt.plot([t_start, t_start + np.max(years)],
                 [-norm.ppf(TestCase.GeneralInfo['Pmax']), -norm.ppf(TestCase.GeneralInfo['Pmax'])],
                 'k--', label='Norm')
        plt.legend()
        plt.title(TestCase.Sections[i].name)
        if not os.path.exists(directory + '\\figures\\' + TestCase.Sections[i].name):
            os.makedirs(directory + '\\figures\\' + TestCase.Sections[i].name + '\\Initial')

        plt.savefig(
            directory + '\\' + 'figures' + '\\' + TestCase.Sections[i].name + '\\Initial\\InitialSituation' + '.png',
            bbox_inches='tight')
        plt.close()

    # plot reliability and failure probability for entire traject:
    figsize = (8,4)
    TestCase.plotReliabilityofDikeTraject(PATH=directory, fig_size=figsize,language=language,flip='off',draw_targetbeta='off',beta_or_prob='beta',outputcsv = True, last=True)
    TestCase.plotReliabilityofDikeTraject(PATH=directory, fig_size=figsize, language=language, flip='off',draw_targetbeta='off', beta_or_prob='prob', last=True)

    print('Finished step 1: assessment of current situation')

    if timing == 1: end = time.time()
    if timing == 1: print("Time elapsed: " + str(end - start) + ' seconds')
    if timing == 1: start = time.time()
    #store stuff:
    if shelves == 1:
        # Save intermediate results to shelf:
        filename = directory + '\\AfterStep1.out'
        # make shelf
        my_shelf = shelve.open(filename, 'n')
        my_shelf['TestCase'] = locals()['TestCase']
        my_shelf.close()

        # open shelf
        # my_shelf = shelve.open(filename)
        # for key in my_shelf:
        #     locals()[key]=my_shelf[key]
        # my_shelf.close()

    ## STEP 2: INITIALIZE AND EVALUATE MEASURES FOR EACH SECTION
    # Result: Measures object with Section name and beta-t-euro relations for each measure
    TestCaseSolutions = {}
    # Calculate for each measure the cost-reliability-time relations
    for i in TestCase.Sections:
        TestCaseSolutions[i.name] = Solutions(i)
        TestCaseSolutions[i.name].fillSolutions(base_dir + '\\' + i.name + '.xlsx')
        TestCaseSolutions[i.name].evaluateSolutions(i, TestCase.GeneralInfo, geometry_plot=False, trange=years,
                                                    plot_dir=directory + '\\figures\\' + i.name + '\\')
        #NB: geometry_plot = 'on' plots the soil reinforcement geometry, but costs a lot of time!
    print('Finished step 2: evaluation of measures')
    if timing == 1: end = time.time()
    if timing == 1: print("Time elapsed: " + str(end - start) + ' seconds')
    if timing == 1: start = time.time()

    #possibly plot beta(t)-cost for all measures at a section:
    if save_beta_measure_plots == 1:
        betaind_array = []
        for i in years: betaind_array.append('beta' + str(i))
        plt_mech = ['Section', 'Piping', 'StabilityInner', 'Overflow']
        for i in TestCase.Sections:
            for betaind in betaind_array:
                for mech in plt_mech:
                    requiredbeta = -norm.ppf(
                        TestCase.GeneralInfo['Pmax'] * (i.Length / TestCase.GeneralInfo['TrajectLength']))
                    plt.figure(1001)
                    TestCaseSolutions[i.name].plotBetaTimeEuro(mechanism=mech, beta_ind=betaind, sectionname=i.name,
                                                               beta_req=requiredbeta)
                    plt.savefig(directory + '\\' + 'figures' + '\\' + i.name + '\\Measures\\' + mech + '_' + betaind + '.png',
                                bbox_inches='tight')
                    plt.close(1001)
        print('Finished making beta plots')

    for i in TestCase.Sections:
        TestCaseSolutions[i.name].SolutionstoDataFrame(filtering='off')

    if timing == 1: end = time.time()
    if timing == 1: print("Time elapsed: " + str(end - start) + ' seconds')
    if timing == 1: start = time.time()
    if shelves == 1:
        # Store intermediate results:
        filename = directory + '\\AfterStep2.out'
        #
        # make shelf
        my_shelf = shelve.open(filename, 'n')
        my_shelf['TestCase'] = locals()['TestCase']
        my_shelf['TestCaseSolutions'] = locals()['TestCaseSolutions']
        my_shelf.close()

    ## STEP 3: EVALUATE THE STRATEGIES
    Strategies = []
    for i in types:
        if i == 'TC':
            # Initialize a strategy type (i.e combination of objective & constraints)
            TestCaseStrategyTC = Strategy('TC')
            # Combine available measures
            TestCaseStrategyTC.combine(TestCase, TestCaseSolutions, filtering='off')
            if timing == 1: end = time.time()
            if timing == 1: print('Combine step for TC')
            if timing == 1: print("Time elapsed: " + str(end - start) + ' seconds')
            # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
            TestCaseStrategyTC.evaluate(TestCase, TestCaseSolutions)

            # plot beta time for all measure steps for each strategy
            TestCaseStrategyTC.plotBetaTime(TestCase, typ='single', path=directory, horizon=np.max(years))
            # plot beta costs for t=0
            plt.figure(101, figsize=(20, 10))
            TestCaseStrategyTC.plotBetaCosts(TestCase, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            plt.close(101)
            # plot beta costs for t=50
            plt.figure(102, figsize=(20, 10))
            TestCaseStrategyTC.plotBetaCosts(TestCase, t=50, path=directory, typ='single', fig_id=101,
                                             horizon=np.max(years))
            TestCaseStrategyTC = replaceNames(TestCaseStrategyTC, TestCaseSolutions)
            plt.close(102)

            # write to csv's
            for i in TestCaseStrategyTC.options: TestCaseStrategyTC.options[i].to_csv(
                directory + '\\results\\' + i + '_Options_TC.csv')
            TestCaseStrategyTC.TakenMeasures.to_csv(directory + '\\results\\' + 'TakenMeasures_TC.csv')

            Strategies.append(TestCaseStrategyTC)

        elif i == 'OI':
            # Initialize a strategy type (i.e combination of objective & constraints)
            TestCaseStrategyOI = Strategy('OI')
            # Combine available measures
            TestCaseStrategyOI.combine(TestCase, TestCaseSolutions, filtering='off',OI_year=OI_year)
            if timing == 1: end = time.time()
            if timing == 1: print('Combine step for OI')
            if timing == 1: print("Time elapsed: " + str(end - start) + ' seconds')
            # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
            TestCaseStrategyOI.evaluate(TestCase, TestCaseSolutions,OI_year=OI_year)
            Strategies.append(TestCaseStrategyOI)

            # plot beta time for all measure steps for each strategy
            TestCaseStrategyOI.plotBetaTime(TestCase, typ='single', path=directory, horizon=np.max(years))
            # plot beta costs for t=0
            plt.figure(101, figsize=(20, 10))
            TestCaseStrategyOI.plotBetaCosts(TestCase, path=directory, typ='single', fig_id=101, horizon=np.max(years))
            plt.close(101)
            # plot beta costs for t=50
            plt.figure(102, figsize=(20, 10))
            TestCaseStrategyOI.plotBetaCosts(TestCase, t=50, path=directory, typ='single', fig_id=101,
                                             horizon=np.max(years))
            TestCaseStrategyOI = replaceNames(TestCaseStrategyOI, TestCaseSolutions)
            plt.close(102)
            # write to csv's
            for i in TestCaseStrategyOI.options: TestCaseStrategyOI.options[i].to_csv(
                directory + '\\results\\' + i + '_Options_OI.csv')
            TestCaseStrategyOI.TakenMeasures.to_csv(directory + '\\results\\' + 'TakenMeasures_OI.csv')



    if shelves == 1:
        # Store final results
        filename = directory + '\\FINALRESULT.out'

        # make shelf
        my_shelf = shelve.open(filename, 'n')
        my_shelf['TestCase'] = locals()['TestCase']
        my_shelf['TestCaseSolutions'] = locals()['TestCaseSolutions']
        if 'TC' in types: my_shelf['TestCaseStrategyTC'] = locals()['TestCaseStrategyTC']
        if 'OI' in types: my_shelf['TestCaseStrategyOI'] = locals()['TestCaseStrategyOI']
        my_shelf.close()

    return Strategies, TestCaseSolutions

def getMeasureTable(Solutions):
    OverallMeasureTable = pd.DataFrame([],columns=['ID','Name'])
    for i in Solutions:
        OverallMeasureTable = pd.concat([OverallMeasureTable,Solutions[i].MeasureTable])
    OverallMeasureTable = OverallMeasureTable.drop_duplicates(subset='ID')
    return OverallMeasureTable


def runNature(strategy,nature,traject,nature_solutions,directory = None, shelves = 1):
    MeasureTable = getMeasureTable(nature_solutions)
    years = list(nature.Probabilities[0].columns)
    nature_orig=copy.deepcopy(nature)
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


def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#this is a more generic function to read and write data from and to a shelve. But it is not implemented fully:
# TODO implement DataAtShelve instead of (un)commenting snippets of code
def DataAtShelve(dir, name, objects = None, mode = 'write'):
    if mode == 'write':
        #make shelf
        my_shelf = shelve.open(dir + '\\' + name, 'n')
        for i in objects.keys():
            my_shelf[i] = objects[i]
        my_shelf.close()
    elif mode == 'read':
        # open shelf
        my_shelf = shelve.open(dir + '\\' + name)
        keys = []
        for key in my_shelf:
            locals()[key]=my_shelf[key]
            keys.append(key)
        my_shelf.close()
        if len(keys) == 1:
            return locals()[keys[0]]