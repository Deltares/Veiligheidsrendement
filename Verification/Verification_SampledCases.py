#IMPORT PACKAGES
from DikeTraject import DikeTraject
from HelperFunctions import DataAtShelve,getMeasureTable, replaceNames, pareto_frontier,readBatchInfo
from Strategy import GreedyStrategy, MixedIntegerStrategy, ParetoFrontier
from Solutions import Solutions
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
def BatchRunOptimization(filepath,LANGUAGE = 'EN', MECHANISMS = ['Overflow', 'StabilityInner', 'Piping'],T = [0, 19, 20, 50, 75, 100],STARTYEAR=2020,plot_on = False,
                         pareto_on=False,StartSet = 0,pareto_sets = 1,pareto_samples=100,run_MIP=True,run_Greedy = True, GreedySettings = {'setting':'cautious','f':1.5,
                                                                                                                   'BCstop':0.1}):

    #Read the data
    TrajectObject = DikeTraject('')
    TrajectObject.ReadAllTrajectInput(filepath, '', T, STARTYEAR)

    #Step 1: Assess current situation
    TrajectObject.runFullAssessment()
    TrajectObject.setProbabilities()
    if plot_on:
        assessment_directory = filepath.joinpath('AssessmentFigures','Initial')
        TrajectObject.plotAssessmentResults(directory=assessment_directory)
        assessment_directory = filepath.joinpath('AssessmentFigures')
        TrajectObject.plotAssessment(PATH=assessment_directory)
        # DataAtShelve(dir=filepath, name='TrajectObject.out', objects={'TrajectObject': TrajectObject}, mode='write')
    #Step 2: Evaluate solutions
    solutions_directory = filepath.joinpath('Solutions')
    SolutionsCollection = {}
    for i in TrajectObject.Sections:
        SolutionsCollection[i.name] = Solutions(i)
        SolutionsCollection[i.name].fillSolutions(filepath.joinpath(i.name + '.xlsx'))
        SolutionsCollection[i.name].evaluateSolutions(i, TrajectObject.GeneralInfo, geometry_plot=False,
                                                      plot_dir=None, preserve_slope =
                                                      True)
        SolutionsCollection[i.name].SolutionstoDataFrame(filtering='off',splitparams=True)
    # SolutionsCollection = DataAtShelve(dir=filepath, name='SolutionsCollection.out', mode='read')

    #Step 3: Greedy search
    if run_Greedy:
        StrategyGreedy = GreedyStrategy('Greedy')
        StrategyGreedy.combine(TrajectObject, SolutionsCollection,splitparams=True)
        if GreedySettings['setting'] == 'cautious':
            StrategyGreedy.evaluate(TrajectObject, SolutionsCollection,splitparams=True,setting="cautious",f_cautious=GreedySettings['f'],BCstop=GreedySettings['BCstop'])
        elif GreedySettings['setting'] == 'robust':
            StrategyGreedy.evaluate(TrajectObject, SolutionsCollection,splitparams=True,setting="robust",BCstop=GreedySettings['BCstop'])

        StrategyGreedy = replaceNames(StrategyGreedy,SolutionsCollection)
        if plot_on:
            StrategyGreedy.plotBetaCosts(TrajectObject,path= filepath,typ='multi',
                                         fig_id=101, symbolmode='on',linecolor='b', labels='Greedy',
                                         MeasureTable=getMeasureTable(SolutionsCollection),beta_or_prob='beta',outputcsv=True,last='yes')

        StrategyGreedy.TakenMeasures.to_csv(filepath.joinpath('TakenMeasures_Greedy.csv'))
        cost_Greedy = StrategyGreedy.determineRiskCostCurve(TrajectObject)
        pd.DataFrame(np.array([cost_Greedy['LCC'],cost_Greedy['TR'],np.add(cost_Greedy['LCC'],cost_Greedy['TR'])]).T,columns=['LCC','TR','TC']).to_csv(filepath.joinpath('TCs_Greedy.csv'),float_format='%.1f')
        StrategyGreedy.makeSolution(filepath.joinpath('TakenMeasures_Optimal_Greedy.csv'),step=cost_Greedy['TC_min']+1)
        StrategyGreedy.makeSolution(filepath.joinpath('TakenMeasures_Final_Greedy.csv'),type='Final')
        f = open(filepath.joinpath('OptimalTC_Greedy.txt'),'w')
        f.write('{:.3f}'.format(cost_Greedy['TC'][cost_Greedy['TC_min']]))
        f.close()
        # DataAtShelve(dir=filepath, name='GreedyResult.out', objects={'StrategyGreedy': StrategyGreedy}, mode='write')

    #Step 4: Mixed Integer Programming
    if run_MIP:
        StrategyMixedInteger = MixedIntegerStrategy('MixedInteger')
        StrategyMixedInteger.combine(TrajectObject,SolutionsCollection,splitparams=True)
        StrategyMixedInteger.make_optimization_input(TrajectObject,SolutionsCollection)
        MixedIntegerModel = StrategyMixedInteger.create_optimization_model()
        MixedIntegerModel.solve()
        MixedIntegerResults = {}
        MixedIntegerResults['Values'] = MixedIntegerModel.solution.get_values()
        MixedIntegerResults['Names'] = MixedIntegerModel.variables.get_names()
        MixedIntegerResults['ObjectiveValue'] = MixedIntegerModel.solution.get_objective_value()
        f = open(filepath.joinpath('OptimalTC_MIP.txt'),'w')
        f.write('{:.3f}'.format(MixedIntegerResults['ObjectiveValue']))
        f.close()
        MixedIntegerResults['Status'] = MixedIntegerModel.solution.get_status_string()
        StrategyMixedInteger.readResults(MixedIntegerResults,dir = filepath,MeasureTable=getMeasureTable(
            SolutionsCollection))

    if pareto_on:
        if not run_MIP:
            StrategyMixedInteger = MixedIntegerStrategy('MixedInteger')
            StrategyMixedInteger.combine(TrajectObject, SolutionsCollection, splitparams=True)
            StrategyMixedInteger.make_optimization_input(TrajectObject, SolutionsCollection)
        StrategyParetoFrontier = ParetoFrontier('ParetoFrontier')
        StrategyParetoFrontier.evaluate(TrajectObject,SolutionsCollection,LCClist=[1,10,50])

        StrategyParetoFrontier.fill_with_MIP(StrategyMixedInteger)
        # StrategyParetoFrontier_unfiltered = copy.deepcopy(StrategyParetoFrontier)
        # StrategyParetoFrontier_unfiltered.evaluate(TrajectObject,SolutionsCollection)
        #with filtering:
        if run_Greedy:
            StrategyParetoFrontier.evaluate(TrajectObject,SolutionsCollection,filepath,NrSets = pareto_sets,StartSet=14,NrSamples=pareto_samples,greedystrategy=StrategyGreedy)
        else:
            StrategyParetoFrontier.evaluate(TrajectObject,SolutionsCollection,filepath,NrSets = pareto_sets,StartSet=14,NrSamples=pareto_samples)

        pass
    # DataAtShelve(dir=filepath, name='MixedIntegerResults.out', objects={'MixedIntegerResults': MixedIntegerResults},mode='write')

def main():
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\OptimizationBatch')
    PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_general_run1')
    # PATH = Path(r'c:\PythonResults\SAFE_results\OptimizationBatch')
    # PATH = Path(r'c:\PythonResults\SAFE_results\batch_results')

    make_input = False
    if make_input:
        print('Make all different cases')
        caselist = readBatchInfo(PATH)
    else:
        caselist = []
        for mainpath in PATH.iterdir():
            if mainpath.is_dir() and mainpath.name != 'BaseData':
                for subpath in PATH.joinpath(mainpath).iterdir():
                    caselist.append((mainpath.name, subpath.name))

    print('Run all computations')
    for i in caselist:
        # if i[0] == '01' and i[1] == '097':
        print('Running case ' + i[0] + 'run ' + i[1])
        BatchRunOptimization(PATH.joinpath(i[0],i[1]))




    #COMPARISON OF TOTAL COST
    fig1,ax1 = plt.subplots()
    ax1.plot(TC_MIP,TC_Greedy,'or')
    ax1.set_xlim(left=np.min(np.floor(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6,right=np.max(np.ceil(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6)
    ax1.set_ylim(bottom=np.min(np.floor(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6,top=np.max(np.ceil(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6)
    coords =[np.min(np.floor(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6, np.max(np.floor(np.divide(np.array([TC_Greedy,TC_MIP]),1e6)))*1e6]
    ax1.plot(coords,coords,'k--')

    mid = np.int32(len(TC_MIP)/2)
    relative = np.sort(np.divide(TC_MIP[mid:], TC_Greedy[mid:]))
    fig3, ax3 = plt.subplots()
    ax3.bar(range(0,mid),relative)

    print('First half deviation of <1% TC : ' + str(np.sum(np.isclose(TC_MIP[0:mid],TC_Greedy[0:mid],rtol=0.01))) + ' out of ' + str(mid) + ' cases')
    print('Second half deviation of <1% TC: ' + str(np.sum(np.isclose(TC_MIP[mid:],TC_Greedy[mid:],rtol=0.01))) + ' out of ' + str(mid) + ' cases')
    print('All cases deviation of <1% TC  : ' + str(np.sum(np.isclose(TC_MIP,TC_Greedy,rtol=0.01))) + ' out of ' + str(len(TC_MIP)) + ' cases')
    plt.show()
    #COMPARISON OF LCC
    fig2,ax2 = plt.subplots()
    ax2.plot(LCC_MIP,LCC_Greedy,'or')
    ax2.set_xlim(left=np.min(np.floor(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6,right=np.max(np.ceil(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6)
    ax2.set_ylim(bottom=np.min(np.floor(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6,top=np.max(np.ceil(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6)
    coords =[np.min(np.floor(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6, np.max(np.floor(np.divide(np.array([LCC_Greedy,LCC_MIP]),1e6)))*1e6]
    ax2.plot(coords,coords,'k--')
    # plt.show()

    mid = np.int32(len(TC_MIP)/2)
    relative = np.sort(np.divide(TC_MIP[mid:], TC_Greedy[mid:]))
    fig4, ax4 = plt.subplots()
    ax4.bar(range(0,mid),relative)
    plt.show()
    print('First half deviation of <1% LCC : ' + str(np.sum(np.isclose(LCC_MIP[0:mid],LCC_Greedy[0:mid],rtol=0.01))) + ' out of ' + str(mid) + ' cases')
    print('Second half deviation of <1% LCC: ' + str(np.sum(np.isclose(LCC_MIP[mid:],LCC_Greedy[mid:],rtol=0.01))) + ' out of ' + str(mid) + ' cases')
    print('All cases deviation of <1% LCC  : ' + str(np.sum(np.isclose(LCC_MIP,LCC_Greedy,rtol=0.01))) + ' out of ' + str(len(LCC_MIP)) + ' cases')


if __name__ == '__main__':
    main()