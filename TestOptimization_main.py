#IMPORT PACKAGES
from DikeClasses import DikeTraject
from HelperFunctions import DataAtShelve,getMeasureTable, replaceNames
from StrategyEvaluation import Solutions, Strategy, SolveMIP

#GENERAL OPTIONS
SHELVES = True

#INDICATE WHICH STEPS TO RUN THROUGH
RUN_STEP_1 = False
RUN_STEP_2 = False
RUN_STEP_3 = True

#INDICATE WHICH METHODS TO EVALUATE
RUN_HEURISTIC              = False
RUN_MULTIINTEGER           = False
RUN_PARETOFRONTIER         = True

#INDICATE THE CASE TO USE
CASE = 'Basic'

#STEP 0: READ GENERAL INPUT
PATH = r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\PaperOptimization\Basic_v2'
LANGUAGE = 'EN'
MECHANISMS = ['Overflow', 'StabilityInner', 'Piping']
T = [0, 19, 20, 50, 75, 100]
STARTYEAR = 2020
TrajectObject = DikeTraject(CASE)
TrajectObject.ReadAllTrajectInput(PATH,CASE,T,STARTYEAR)

#STEP 1: DO ASSESSMENT
if RUN_STEP_1:
    TrajectObject.runFullAssessment()

    assessment_directory = PATH + '\\AssessmentFigures\\Initial'
    TrajectObject.plotAssessmentResults(directory=assessment_directory)

    assessment_directory = PATH + '\\AssessmentFigures'
    TrajectObject.plotReliabilityofDikeTraject(PATH=assessment_directory)
    print('Step 1: assessment finished')
    #store results:
    if SHELVES: DataAtShelve(dir = PATH, name = 'Step1Result.out', objects = {'TrajectObject': TrajectObject}, mode='write')
else:
    TrajectObject = DataAtShelve(dir = PATH, name = 'Step1Result.out', mode='read')

#STEP 2: EVALUATE ALL MEASURES
if RUN_STEP_2:
    solutions_directory = PATH + '\\Solutions'
    SolutionsCollection = {}
    for i in TrajectObject.Sections:
        SolutionsCollection[i.name] = Solutions(i)
        SolutionsCollection[i.name].fillSolutions(PATH + '\\' + i.name + '.xlsx')
        SolutionsCollection[i.name].evaluateSolutions(i, TrajectObject.GeneralInfo, geometry_plot=False, plot_dir=solutions_directory + '\\' +i.name, preserve_slope = True)
        SolutionsCollection[i.name].SolutionstoDataFrame(filtering='off')
    print('Step 2: evaluation of measures finished')
    #store results:
    if SHELVES: DataAtShelve(dir = PATH, name = 'Step2Result.out', objects = {'SolutionsCollection': SolutionsCollection}, mode = 'write')
else:
    SolutionsCollection = DataAtShelve(dir = PATH, name = 'Step2Result.out', mode = 'read')

#store to an accessible format

#STEP 3: FIND SOLUTIONS FOR ALL SELECTED METHODS
if RUN_HEURISTIC:
    StrategyHeuristic = Strategy('Heuristic')
    StrategyHeuristic.combine(TrajectObject, SolutionsCollection)
    StrategyHeuristic.evaluate(TrajectObject, SolutionsCollection)

    StrategyHeuristic = replaceNames(StrategyHeuristic,SolutionsCollection)
    StrategyHeuristic.plotBetaCosts(TrajectObject,path= PATH + '\\Solutions',typ='multi',
                                     fig_id=101, symbolmode='on',linecolor='b', labels='Heuristic',
                                     MeasureTable=getMeasureTable(SolutionsCollection),beta_or_prob='beta',outputcsv=True,last='yes')
    # write to csv's
    for i in StrategyHeuristic.options: StrategyHeuristic.options[i].to_csv(
        PATH + '\\Solutions\\' + i + '_Options.csv')
    StrategyHeuristic.TakenMeasures.to_csv(PATH + '\\TakenMeasures_Heuristic.csv')
    if SHELVES: DataAtShelve(dir = PATH, name = 'HeuristicResult.out', objects = {'StrategyHeuristic': StrategyHeuristic}, mode = 'write')
if RUN_MULTIINTEGER:
    StrategyMultiInteger = Strategy('MultiInteger')
    StrategyMultiInteger.combine(TrajectObject,SolutionsCollection)
    StrategyMultiInteger.make_optimization_input(TrajectObject,SolutionsCollection)
    MultiIntegerModel = StrategyMultiInteger.create_optimization_model()
    MultiIntegerSolution = SolveMIP(MultiIntegerModel)

    StrategyMultiInteger.readResults(MultiIntegerModel)
    # StrategyMultiInteger.checkConstraintSatisfaction(MultiIntegerModel)
    if SHELVES: DataAtShelve(dir = PATH, name = 'MultiIntegerResult.out', objects = {'StrategyMultiInteger': StrategyMultiInteger}, mode = 'write')

    print()

if RUN_PARETOFRONTIER:
    StrategyParetoFrontier = Strategy('ParetoFrontier')
    StrategyParetoFrontier.combine(TrajectObject,SolutionsCollection)
    #first step:
    StrategyParetoFrontier.filter('ParetoPerSection')
    #filter out as many alternatives as possible based on the cost-risk ratio for each section
    #second step:
    #brute force all remaining alternatives
    #third step:
    #determine pareto front

    if SHELVES: DataAtShelve(dir = PATH, name = 'ParetoFrontierResult.out', objects = {'StrategyParetoFrontier': StrategyParetoFrontier}, mode = 'write')
    print()
# if RUN_SIMULATEDANNEALING:
#     pass
#     print()

## STEP 4: POSTPROCESSING OF RESULTS