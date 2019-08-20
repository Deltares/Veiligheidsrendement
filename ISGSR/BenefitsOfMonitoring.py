# This is the main script used for the GeoRisk 2019 paper.
# Goal: assess what a better assessment of permeability will give in terms of cVoI
# We consider a 5 section dike reach based on 16-4.
#The time line is: we have time before the reinforcement starts at t = i. The time available from
#t=0 to t=i is sufficient to get a new estimate of k.
# So the actual analysis starts at t=i

import matplotlib.pyplot as plt
from DikeClasses_old import DikeSection, LoadInput, MechanismReliabilityCollection, DikeTraject
from StrategyEvaluation import Solutions, Strategy
from HelperFunctions import calc_r_exit, adaptInput
from RunModel import runFullModel runNature
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
language = 'EN'
if timing == 1: start = time.time()
if timing == 1: start0 = time.time()

## MAKE TRAJECT OBJECT
TestCaseGeoRisk = DikeTraject('TestCaseGeoRisk','16-4')

## READ ALL DATA


print('Start creating all the files and folders')
TestCaseGeoRisk.ReadAllTrajectInput(pad, years0)

monitored_sections = ['DV02']
grid_data_low = pd.read_csv(pad + "\\scenario_k1736.csv")
grid_data_high = pd.read_csv(pad + "\\scenario_k3472.csv")


#determine if the section has a high or low conductivity
CasesGeoRisk = []   # First case is the base case
if len(monitored_sections) == 1:
    if monitored_sections[0] in ['DV02','DV03']:
        CasesGeoRisk = adaptInput(grid_data_low,monitored_sections,copy.deepcopy(TestCaseGeoRisk))
    else:
        CasesGeoRisk = adaptInput(grid_data_high, monitored_sections, copy.deepcopy(TestCaseGeoRisk))
else:
    pass
    #to be done if we want to include system monitoring. We should then combine the cases.

CasesGeoRisk = CasesGeoRisk
BaseCaseStrategy,BaseCaseSolutions = runFullModel(TestCaseGeoRisk, 'base', pad, mechanisms=mechanisms, years=years0, timing=timing, save_beta_measure_plots=save_beta_measure_plots,language='EN',types=['TC'])
CasesGeoRisk_no_monitoring = []
## STEP 1: SAFETY ASSESSMENT
for i in range(0,len(CasesGeoRisk)):
    BeliefCaseStrategy,BeliefCaseSolutions = runFullModel(CasesGeoRisk[i], i, pad, mechanisms=mechanisms, years=years0, timing=timing, save_beta_measure_plots=save_beta_measure_plots,language='EN',types=['TC'])
    print()
    NoMonitoringResult = runNature(BaseCaseStrategy[0].TakenMeasures,BeliefCaseStrategy[0],TestCaseGeoRisk,BeliefCaseSolutions,
              directory=pad+'\\Case_' + str(i) + '_base')
    CasesGeoRisk_no_monitoring.append(copy.deepcopy(NoMonitoringResult))
if timing == 1: end = time.time()
if timing == 1: print("time elapsed: " + str(end-start) + ' seconds')
if timing == 1: start = time.time()
if timing == 1: end = time.time()
if timing == 1: print("Overall time elapsed: " + str(end-start0) + ' seconds')


#We make for each case that was run a new StrategyObject with the solutions from the case, the base situation from the case, and the measures from the prior situation/strategy
#We evaluate that StrategyObject with as given set of measures TakeMeasures_Case
#Thus we determine the probabilities after every step
