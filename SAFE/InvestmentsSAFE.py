## This script can calculate life-cycle reliability and costs for all measures for various mechanisms






import matplotlib.pyplot as plt
from DikeClasses import DikeTraject
from HelperFunctions import runModel, getMeasureTable
from StrategyEvaluation import calcTrajectProb
import time
import shelve
import pandas as pd
import os
from scipy.stats import norm

## GENERAL SETTINGS
timing = 1
traject= '16-4'
save_beta_measure_plots = 0
years0= [0, 19, 20, 50, 75, 100]
mechanisms = ['Overflow', 'StabilityInner', 'Piping']
# pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\SAFE_' + traject + '_LE'
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\Dijkwerkersdag_aangepast'
language = 'NL'
if timing == 1: start = time.time()
if timing == 1: start0 = time.time()

## MAKE TRAJECT OBJECT
TestCase = DikeTraject('TestCase',traject)



## Run the model
casename = '2025'
directory = pad + '\\Case_' + casename
OI_year = 0
## READ ALL DATA
##First we read all the input data for the different sections. We store these in a Traject object.
#Initialize a list of all sections that are of relevance (these start with DV).
print('Start creating all the files and folders')
TestCase.ReadAllTrajectInput(pad, 'Case_' + casename, years0)


# filename = directory + '\\AfterStep2'
#
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     locals()[key] = my_shelf[key]
# my_shelf.close()
#
# betaind_array = []
# for i in years0: betaind_array.append(i)
# plt_mech = ['Section', 'Piping', 'StabilityInner', 'Overflow']
# for i in TestCase.Sections:
#     for betaind in betaind_array:
#         for mech in plt_mech:
#             requiredbeta = -norm.ppf(
#                 TestCase.GeneralInfo['Pmax'] * (i.Length / TestCase.GeneralInfo['TrajectLength']))
#             plt.figure(1001)
#             TestCaseSolutions[i.name].plotBetaTimeEuro(mechanism=mech, beta_ind=betaind, sectionname=i.name,
#                                                        beta_req=requiredbeta)
#             plt.savefig(directory + '\\' + 'figures' + '\\' + i.name + '\\Measures\\' + mech + '_' + str(betaind+2025) + '.png',
#                         bbox_inches='tight')
#             plt.close(1001)


# AllStrategies, Solutions = runModel(TestCase, casename, pad, years=years0, timing=timing, save_beta_measure_plots=save_beta_measure_plots,language='NL',types=['TC','OI'],OI_year = OI_year) #,TestCaseSolutions=TestCaseSolutions)
#

filename = pad + '\\Case_' + casename + '\\FINALRESULT.out'
# open shelf
my_shelf = shelve.open(filename)
for key in my_shelf:
    locals()[key]=my_shelf[key]
my_shelf.close()
MeasureTable = getMeasureTable(TestCaseSolutions)

#
#Plot the beta-t:
beta_t = []; step = 0

#plot beta time for all measure steps for each strategy
setting = ['beta','prob']
#plot beta costs for t=0
figure_size = (12,7)
for i in setting:
    # LCC PLOTS
    plt.figure(101,figsize=figure_size)
    TestCaseStrategyTC.plotBetaCosts(TestCase,path= pad + '\\Case_' + casename + '\\figures',typ='multi',
                                     fig_id=101, symbolmode='on',linecolor='b', labels='TC',
                                     MeasureTable=MeasureTable,beta_or_prob=i,outputcsv=True)
    TestCaseStrategyOI.plotBetaCosts(TestCase,path= pad + '\\Case_' + casename + '\\figures',typ='multi',
                                     fig_id=101,last='yes', symbolmode='on',labels='OI',
                                     MeasureTable=MeasureTable,beta_or_prob=i,outputcsv=True)

    #plot beta costs for t=50
    plt.figure(102,figsize=figure_size)
    TestCaseStrategyOI.plotBetaCosts(TestCase,t = 50, path= pad + '\\Case_' + casename + '\\figures',
                                     typ='multi',fig_id=102,symbolmode='on',labels='OI',
                                     MeasureTable=MeasureTable,beta_or_prob=i,outputcsv=True)
    TestCaseStrategyTC.plotBetaCosts(TestCase,t = 50, path= pad + '\\Case_' + casename + '\\figures',
                                     typ='multi',fig_id=102,symbolmode='on',linecolor='b', labels='TC',
                                     MeasureTable=MeasureTable,last='yes',beta_or_prob=i,outputcsv=True)

    #INITIAL COSTS PLOT
    plt.figure(103,figsize=figure_size)
    TestCaseStrategyTC.plotBetaCosts(TestCase,cost_type = 'Initial', path= pad + '\\Case_' + casename + '\\figures',
                                     typ='multi',fig_id=103, symbolmode='on',linecolor='b', labels='TC',
                                     MeasureTable=MeasureTable,beta_or_prob=i,outputcsv=True)
    TestCaseStrategyOI.plotBetaCosts(TestCase,cost_type = 'Initial', path= pad + '\\Case_' + casename + '\\figures',
                                     typ='multi',fig_id=103,last='yes', symbolmode='on',labels='OI',
                                     MeasureTable=MeasureTable,beta_or_prob=i,outputcsv=True)

if not 'TC' in os.listdir(pad + '\\Case_' + casename): os.makedirs(pad + '\\Case_' + casename + '\\TC')
if not 'OI' in os.listdir(pad + '\\Case_' + casename): os.makedirs(pad + '\\Case_' + casename + '\\OI')
TestCaseStrategyTC.plotInvestmentSteps(TestCase, path= pad + '\\Case_' + casename + '\\TC',figure_size = (6,4))
TestCaseStrategyOI.plotInvestmentSteps(TestCase, path= pad + '\\Case_' + casename + '\\OI',figure_size = (6,4))

## LOG of all probabilities for all steps:
TestCaseStrategyOI.writeProbabilitiesCSV(path= pad + '\\Case_' + casename + '\\OI', type='OI')
TestCaseStrategyTC.writeProbabilitiesCSV(path= pad + '\\Case_' + casename + '\\TC', type='TC')
ps = []
for i in TestCaseStrategyOI.Probabilities:
    beta_t, p_t = calcTrajectProb(i,horizon=100)
    ps.append(p_t)
ps = pd.DataFrame(ps,columns=range(0,101))
ps.to_csv(path_or_buf=pad + '\\Case_' + casename + '\\OI\\PfT_OI.csv')
ps = []
for i in TestCaseStrategyTC.Probabilities:
    beta_t, p_t = calcTrajectProb(i,horizon=100)
    ps.append(p_t)
ps = pd.DataFrame(ps,columns=range(0,101))
ps.to_csv(path_or_buf=pad + '\\Case_' + casename + '\\TC\\PfT_TC.csv')

if timing == 1: end = time.time()
if timing == 1: print("time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()
if timing == 1: end = time.time()
if timing == 1: print("Overall time elapsed: " + str(end-start0) + ' seconds')