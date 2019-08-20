# This is the main script used for the GeoRisk 2019 paper.
# Goal: assess what a better assessment of permeability will give in terms of cVoI
# We consider a 5 section dike reach based on 16-4.
#The time line is: we have time before the reinforcement starts at t = i. The time available from
#t=0 to t=i is sufficient to get a new estimate of k.
# So the actual analysis starts at t=i

import matplotlib.pyplot as plt
from DikeClasses_old import DikeSection, LoadInput, MechanismReliabilityCollection, DikeTraject
from StrategyEvaluation import Solutions, Strategy
import os
from scipy.stats import norm
import numpy as np
import copy
import pandas as pd
import time
import shelve

## GENERAL SETTINGS
timing = 1
save_beta_measure_plots = 0
years0= [0, 1, 10, 20, 40, 50]
mechanisms = ['Overflow', 'StabilityInner', 'Piping']
pad = 'd:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\98_Papers\\Conference\\GeoRisk_2019\\Calculations\\Input'

if timing == 1: start = time.time()
if timing == 1: start0 = time.time()

## MAKE TRAJECT OBJECT
TestCaseGeoRisk = DikeTraject('TestCaseGeoRisk','16-4')

## READ ALL DATA


print('Start creating all the files and folders')
TestCaseGeoRisk.ReadAllTrajectInput(pad, years0)

## STEP 1: SAFETY ASSESSMENT


print('Start step 1: safety assessment')
for i in range(0,len(TestCaseGeoRisk.Sections)):
    for j in mechanisms:
        TestCaseGeoRisk.Sections[i].Reliability.Mechanisms[j].generateLCRProfile(TestCaseGeoRisk.Sections[i].Reliability.Load,mechanism= j,trajectinfo=TestCaseGeoRisk.GeneralInfo)
    TestCaseGeoRisk.Sections[i].Reliability.calcSectionReliability(TrajectInfo=TestCaseGeoRisk.GeneralInfo,length=TestCaseGeoRisk.Sections[i].Length)
    plt.figure(1)
    [TestCaseGeoRisk.Sections[i].Reliability.Mechanisms[j].drawLCR(label=j,type='Standard',tstart=2025) for j in mechanisms]
    plt.plot([2025, 2025+np.max(years0)], [-norm.ppf(TestCaseGeoRisk.GeneralInfo['Pmax']), -norm.ppf(TestCaseGeoRisk.GeneralInfo['Pmax'])], 'k--', label = 'Norm')
    plt.legend()
    plt.title(TestCaseGeoRisk.Sections[i].name)
    plt.savefig(pad + '\\' + 'figures' + '\\' + TestCaseGeoRisk.Sections[i].name + '\\Initial\\InitialSituation' + '.png', bbox_inches='tight')
    plt.close()

#plot for entire traject
TestCaseGeoRisk.plotReliabilityofDikeTraject(pathname=pad,fig_size=(14,6))

print('Finished step 1: assessment of current situation')

if timing == 1: end = time.time()
if timing == 1: print("Time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()

# Save intermediate results to shelf:
filename = pad + '\\AfterStep1.out'

#make shelf
my_shelf = shelve.open(filename,'n')
my_shelf['TestCaseGeoRisk'] = globals()['TestCaseGeoRisk']
my_shelf.close()

#open shelf
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

## INITIALIZE MEASURES FOR EACH SECTION
# Result: Measures object with Section name and beta-t-euro relations for each measure
TestCaseSolutions = {}
# Calculate for each measure the cost-reliability-time relations
for i in TestCaseGeoRisk.Sections:
    TestCaseSolutions[i.name] = Solutions(i)
    TestCaseSolutions[i.name].fillSolutions(pad + '\\' + i.name + '.xlsx')
    TestCaseSolutions[i.name].evaluateSolutions(i,TestCaseGeoRisk.GeneralInfo, geometry_plot='off',trange=years0,plot_dir = pad + '\\figures\\' + i.name + '\\')
print('Finished step 2: evaluation of measures')
if timing == 1: end = time.time()
if timing == 1: print("Time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()

if save_beta_measure_plots == 1:
    betaind_array = []
    for i in years0: betaind_array.append('beta' + str(i))
    plt_mech = ['Section', 'Piping', 'StabilityInner', 'Overflow']
    for i in TestCaseGeoRisk.Sections:
        for betaind in betaind_array:
            for mech in plt_mech:
                requiredbeta = -norm.ppf(TestCaseGeoRisk.GeneralInfo['Pmax'] * (i.Length / TestCaseGeoRisk.GeneralInfo['TrajectLength']))
                plt.figure(1001)
                TestCaseSolutions[i.name].plotBetaTimeEuro(mechanism=mech,beta_ind = betaind, sectionname=i.name,beta_req=requiredbeta)
                plt.savefig(pad + '\\' + 'figures' + '\\' + i.name + '\\Measures\\' + mech + '_' + betaind + '.png', bbox_inches='tight')
                plt.close(1001)
    print('Finished making beta plots')

for i in TestCaseGeoRisk.Sections:
    TestCaseSolutions[i.name].SolutionstoDataFrame(filtering='on')

if timing == 1: end = time.time()
if timing == 1: print("Time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()

#Store intermediate results:
filename = pad + '\\AfterStep2.out'
#
# make shelf
my_shelf = shelve.open(filename,'n')
my_shelf['TestCaseGeoRisk'] = globals()['TestCaseGeoRisk']
my_shelf['TestCaseSolutions'] = globals()['TestCaseSolutions']
my_shelf.close()

#open shelf
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

## STEP 3: EVALUATE THE STRATEGIES
TestCaseGeoRisk.GeneralInfo['FloodDamage'] = 23e9   #add flood damage (do this in the beginning)

# Initialize a strategy type (i.e combination of objective & constraints)
TestCaseStrategyTC = Strategy('TC')



# Combine available measures
TestCaseStrategyTC.combine(TestCaseGeoRisk,TestCaseSolutions,filtering = 'on')



if timing == 1: end = time.time()
if timing == 1: print('Combine step for TC')
if timing == 1: print("Time elapsed: " + str(end-start) + ' seconds')

# Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
TestCaseStrategyTC.evaluate(TestCaseGeoRisk,TestCaseSolutions)



# Store final results
filename = pad + '\\FINALRESULT.out'

# make shelf
my_shelf = shelve.open(filename,'n')
my_shelf['TestCaseGeoRisk'] = globals()['TestCaseGeoRisk']
my_shelf['TestCaseSolutions'] = globals()['TestCaseSolutions']
my_shelf['TestCaseStrategyTC'] = globals()['TestCaseStrategyTC']
my_shelf.close()



#open shelf
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

#Plot the beta-t:
beta_t = []; step = 0


#plot beta time for all measure steps for each strategy
TestCaseStrategyTC.plotBetaTime(TestCaseGeoRisk,typ = 'single',path=pad, horizon=np.max(years0))



#plot beta costs for t=0
plt.figure(101,figsize=(20,10))
TestCaseStrategyTC.plotBetaCosts(TestCaseGeoRisk,path= pad,typ='single',fig_id=101, horizon=np.max(years0))



#plot beta costs for t=50
plt.figure(102,figsize=(20,10))
TestCaseStrategyTC.plotBetaCosts(TestCaseGeoRisk,t = 50, path= pad,typ='single',fig_id=101, horizon=np.max(years0))



def replaceNames(TestCaseStrategy, TestCaseSolutions):
    TestCaseStrategy.TakenMeasures =TestCaseStrategy.TakenMeasures.reset_index(drop=True)
    for i in range(1, len(TestCaseStrategy.TakenMeasures)):
        names = TestCaseStrategy.TakenMeasures.iloc[i]['name']
        if isinstance(names, list):
            for j in range(0, len(names)):
                names[j] = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names[j]].parameters['Name']
        else:
            names = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names].parameters['Name']
        TestCaseStrategy.TakenMeasures.at[i, 'name'] = names
    return TestCaseStrategy

TestCaseStrategyTC = replaceNames(TestCaseStrategyTC,TestCaseSolutions)



#write to csv's
for i in TestCaseStrategyTC.options: TestCaseStrategyTC.options[i].to_csv(pad + '\\results\\' + i + '_Options_TC.csv')
TestCaseStrategyTC.TakenMeasures.to_csv(pad + '\\results\\' + 'TakenMeasures_TC.csv')







if timing == 1: end = time.time()
if timing == 1: print("time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()
if timing == 1: end = time.time()
if timing == 1: print("Overall time elapsed: " + str(end-start0) + ' seconds')


# We loop over different input sets of k and r_exit
# Pick a section

#create list of TrajectObjects with changed inputs for k and r_exit. k is determined from the distribution of representative values.
# calculate for each one
print()
    # For each set we consider the assessment given the values of k (and r_exit)

    # We determine the available measures and their life-cycle reliability

    # We optimize total costs for the dike reach

    # We save the results in a to be decided format so that we have all runs together