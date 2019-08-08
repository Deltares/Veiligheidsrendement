#IMPORT PACKAGES
from DikeClasses import DikeTraject
from HelperFunctions import DataAtShelve,getMeasureTable, replaceNames, pareto_frontier
from StrategyEvaluation import Solutions, Strategy, SolveMIP
from pathlib import Path
import copy
import pandas
#GENERAL OPTIONS
SHELVES = True

#INDICATE WHICH STEPS TO RUN THROUGH
RUN_STEP_1 = False
RUN_STEP_2 = False
RUN_STEP_3 = True

#INDICATE WHICH METHODS TO EVALUATE
RUN_HEURISTIC              = True
RUN_MIXEDINTEGER           = True
RUN_PARETOFRONTIER         = True
READ_MIXEDINTEGER          = True
#INDICATE THE CASE TO USE
CASE = 'Basic_NoDiaphragm'

#STEP 0: READ GENERAL INPUT
PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland ' \
         'SAFE\WJKlerk\SAFE\data\PaperOptimization\Basic')
LANGUAGE = 'EN'
MECHANISMS = ['Overflow', 'StabilityInner', 'Piping']
T = [0, 19, 20, 50, 75, 100]
STARTYEAR = 2020
TrajectObject = DikeTraject(CASE)
TrajectObject.ReadAllTrajectInput(PATH,CASE,T,STARTYEAR)

#STEP 1: DO ASSESSMENT
if RUN_STEP_1:
    TrajectObject.runFullAssessment()

    assessment_directory = PATH.joinpath('AssessmentFigures').joinpath('Initial')
    TrajectObject.plotAssessmentResults(directory=assessment_directory)

    assessment_directory = PATH.joinpath('AssessmentFigures')
    TrajectObject.plotReliabilityofDikeTraject(PATH=assessment_directory)
    print('Step 1: assessment finished')
    #store results:
    if SHELVES: DataAtShelve(dir = PATH, name = 'Step1Result.out', objects = {'TrajectObject': TrajectObject}, mode='write')
else:
    TrajectObject = DataAtShelve(dir = PATH, name = 'Step1Result.out', mode='read')

#STEP 2: EVALUATE ALL MEASURES
if RUN_STEP_2:
    solutions_directory = PATH.joinpath('Solutions')
    SolutionsCollection = {}
    for i in TrajectObject.Sections:
        SolutionsCollection[i.name] = Solutions(i)
        SolutionsCollection[i.name].fillSolutions(PATH.joinpath(i.name + '.xlsx'))
        SolutionsCollection[i.name].evaluateSolutions(i, TrajectObject.GeneralInfo, geometry_plot=False,
                                                      plot_dir=solutions_directory.joinpath(i.name), preserve_slope =
                                                      True)
        SolutionsCollection[i.name].SolutionstoDataFrame(filtering='off',splitparams=True)
    print('Step 2: evaluation of measures finished')
    #store results:
    if SHELVES: DataAtShelve(dir = PATH, name = 'Step2Result.out', objects = {'SolutionsCollection': SolutionsCollection}, mode = 'write')
else:
    SolutionsCollection = DataAtShelve(dir = PATH, name = 'Step2Result.out', mode = 'read')

#store to an accessible format

#STEP 3: FIND SOLUTIONS FOR ALL SELECTED METHODS
if RUN_HEURISTIC:
    StrategyHeuristic = Strategy('Heuristic')
    StrategyHeuristic.combine(TrajectObject, SolutionsCollection,splitparams=True)
    StrategyHeuristic.evaluate(TrajectObject, SolutionsCollection,splitparams=True)

    StrategyHeuristic = replaceNames(StrategyHeuristic,SolutionsCollection)
    StrategyHeuristic.plotBetaCosts(TrajectObject,path= PATH.joinpath('Solutions'),typ='multi',
                                     fig_id=101, symbolmode='on',linecolor='b', labels='Heuristic',
                                     MeasureTable=getMeasureTable(SolutionsCollection),beta_or_prob='beta',outputcsv=True,last='yes')
    # write to csv's
    for i in StrategyHeuristic.options: StrategyHeuristic.options[i].to_csv(
        PATH.joinpath('Solutions').joinpath(i + '_Options.csv'))
    StrategyHeuristic.TakenMeasures.to_csv(PATH.joinpath(CASE).joinpath('TakenMeasures_Heuristic.csv'))
    if SHELVES: DataAtShelve(dir = PATH, name = 'HeuristicResult.out', objects = {'StrategyHeuristic': StrategyHeuristic}, mode = 'write')
if RUN_MIXEDINTEGER:
    StrategyMixedInteger = Strategy('MixedInteger')
    StrategyMixedInteger.combine(TrajectObject,SolutionsCollection,splitparams=True)
    StrategyMixedInteger.make_optimization_input(TrajectObject,SolutionsCollection)
    MixedIntegerModel = StrategyMixedInteger.create_optimization_model()
    MixedIntegerSolution = SolveMIP(MixedIntegerModel)

    # StrategyMixedInteger.readResults(MixedIntegerModel)
    # StrategyMixedInteger.checkConstraintSatisfaction(MixedIntegerModel)
    if SHELVES:
        DataAtShelve(dir = PATH, name = 'MixedIntegerStrategy.out', objects = {'StrategyMixedInteger':
                                                                            StrategyMixedInteger}, mode = 'write')
        MixedIntegerResults = {}
        MixedIntegerResults['Values']= MixedIntegerModel.solution.get_values()
        MixedIntegerResults['Names']= MixedIntegerModel.variables.get_names()
        MixedIntegerResults['ObjectiveValue'] = MixedIntegerModel.solution.get_objective_value()
        MixedIntegerResults['Status'] = MixedIntegerModel.solution.get_status_string()

        DataAtShelve(dir = PATH, name = 'MixedIntegerResults.out', objects = {'MixedIntegerResults': MixedIntegerResults},
                     mode = 'write')

    print()

if RUN_PARETOFRONTIER:
    StrategyParetoFrontier = Strategy('ParetoFrontier')
    StrategyParetoFrontier.combine(TrajectObject,SolutionsCollection,splitparams=True)


    # StrategyParetoFrontier_unfiltered = copy.deepcopy(StrategyParetoFrontier)
    # StrategyParetoFrontier_unfiltered.evaluate(TrajectObject,SolutionsCollection)

    #with filtering:
    StrategyParetoFrontier.filter(TrajectObject,'ParetoPerSection')
    import cProfile
    # cProfile.run('StrategyParetoFrontier.evaluate(TrajectObject,SolutionsCollection)')
    StrategyParetoFrontier.evaluate(TrajectObject,SolutionsCollection)


    if SHELVES: DataAtShelve(dir = PATH, name = 'ParetoFrontierResult.out', objects = {'StrategyParetoFrontier': StrategyParetoFrontier}, mode = 'write')
    print()
# if RUN_SIMULATEDANNEALING: or DIJKSTRA SHORTEST PATH/UNIFORM COST SEARCH
#     pass
#     print()

## STEP 4: POSTPROCESSING OF RESULTS

if READ_MIXEDINTEGER:
    StrategyMixedInteger = DataAtShelve(dir = PATH, name = 'MixedIntegerStrategy.out', mode = 'read')
    MixedIntegerResults= DataAtShelve(dir = PATH, name = 'MixedIntegerResults.out', mode = 'read')
    StrategyMixedInteger.readResults(MixedIntegerResults,dir = PATH.joinpath(CASE),MeasureTable=getMeasureTable(
        SolutionsCollection))


print()
import matplotlib.pyplot as plt
import numpy as np

plt.figure(1000,figsize=(8,8))
#plot the risk-cost relation:
if 'StrategyParetoFrontier' in locals():
    # [LCC_Pareto, TR_Pareto] = [StrategyParetoFrontier_unfiltered.LCC_combis, StrategyParetoFrontier_unfiltered.TotalRisk_combis]
    # [LCC_ParetoFront,TR_ParetoFront,index_all] = pareto_frontier(LCC_Pareto,TR_Pareto,maxX=False,maxY=False)
    # plt.plot(LCC_Pareto,TR_Pareto,linestyle = '',marker = 'o',color = 'plum',label='All Runs')
    # plt.plot(LCC_ParetoFront,TR_ParetoFront,linestyle = '',marker = 's',color = 'indigo',label='Pareto Front')

    [LCC_Pareto, TR_Pareto] = [StrategyParetoFrontier.LCC_combis, StrategyParetoFrontier.TotalRisk_combis]
    [LCC_ParetoFront,TR_ParetoFront,index] = pareto_frontier(LCC_Pareto,TR_Pareto,maxX=False,maxY=False)
    plt.plot(LCC_Pareto,TR_Pareto,linestyle = '',marker = 'o',color = 'lightgray',label='All Runs')
    plt.plot(LCC_ParetoFront,TR_ParetoFront,linestyle = '-',marker = 'd',color = 'dimgray',label='Pareto Front')

if 'StrategyHeuristic' in locals():
    [TR_Heuristic, LCC_Heuristic] = StrategyHeuristic.determineRiskCostCurve(TrajectObject,PATH.joinpath(
        CASE).joinpath('PfDumps'))
    plt.plot(LCC_Heuristic,TR_Heuristic, '-or', label='Heuristic')

if 'StrategyMixedInteger' in locals():
    [TR_MixedInteger, LCC_MixedInteger] = StrategyMixedInteger.determineRiskCostCurve(TrajectObject,PATH.joinpath(
        CASE).joinpath('PfDumps'))
    plt.plot(LCC_MixedInteger,TR_MixedInteger,'ob',label='Mixed Integer')

    #make radius line for optimum:
    total = TR_MixedInteger+LCC_MixedInteger
    x = np.arange(0,total,1)
    y = total - x
    plt.plot(x,y,linestyle=':',color='blue',label='Radius of MIP optimum')
plt.ylabel('Total Risk')
plt.xlabel('LCC')
plt.yscale('log')
plt.ylim((1e4,5e7))
plt.xlim((0,1.2e8))
# plt.xlim((0,2e7))
plt.legend(loc=1)
plt.title('Comparison of risk and cost for different methods')
plt.savefig(PATH.joinpath(CASE).joinpath('ComparisonOfMethods_logscale.png'),dpi=300, bbox_inches='tight')
# plt.savefig(PATH.joinpath(CASE).joinpath('ComparisonOfMethods.png'),dpi=300, bbox_inches='tight')
plt.close()
#plot with all TC of all draws in bar graph and the final solution of MIP and Heuristic:
TC_Pareto = np.sort(np.add(LCC_Pareto,TR_Pareto))
TC_ParetoFront = np.add(LCC_ParetoFront,TR_ParetoFront)
TC_Heuristic =np.add(LCC_Heuristic,TR_Heuristic)
TC_MIP = np.add(LCC_MixedInteger,TR_MixedInteger)

plt.figure(1001,figsize=(8,6))
plt.plot(range(0,len(TC_Pareto)),TC_Pareto,color='lightgray',label='individual runs')
plt.axhline(y=TC_MIP,xmin=0,xmax=len(TC_Pareto),color='r',linestyle='--',label='Mixed Integer Optimum')
plt.hlines(y=TC_Heuristic,xmin=0,xmax=len(TC_Pareto),color='g',linestyle='--',label='Greedy heuristic steps')
plt.axhline(y=TC_Heuristic[-1],xmin=0,xmax=len(TC_Pareto),color='g',linestyle='-',label='Greedy heuristic '
                                                                                                'Optimum')
plt.xlim((0,len(TC_Pareto)))
plt.legend(loc=2)
plt.title('Comparison of total cost for different methods')
plt.savefig(PATH.joinpath(CASE).joinpath('Methods.png'),dpi=300, bbox_inches='tight')
print(TC_Heuristic)
print(TC_MIP)


#pseudocode:
#if type = heuristic
#loop over takenmeasures
    #implemnet option
    #calculate risk
    #save (Total Risk, LCC)
#elif type = MIP
#implemnet options
#calculate risk
#save (total risk, lcc)
#plot all points