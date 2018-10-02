## This script can calculate life-cycle reliability and costs for all measures for various mechanisms
import matplotlib.pyplot as plt
from DikeClasses import DikeSection, MechanismInput, LoadInput, MechanismReliabilityCollection, Strategy, InitialSituation, StrategyCalculation

import numpy as np
import time
import copy
start = time.time()

#We read a section
#For now we make two dummy sections
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-4\input'
inputfile1 = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\TestCase\Section1.xlsx'

Section1 = DikeSection('DV1','16-4')
Section1.readGeneralInfo(inputfile1,'General')
Section1.Reliability.Load = LoadInput()
Section1.Reliability.Load.set_fromDesignTable(r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\TestCase\DV1\DesignTable.txt')
Section1.Reliability.Load.set_annual_change(type='gamma',parameters=[0.01,0.005,0.0])
# Section1.Reliability.Load.plot_load_cdf()


## ANALYSIS FOR OVERFLOW
# print('Analysis for overflow for section 1')
# Section1.Reliability.Overflow = MechanismReliabilityCollection('Overflow','Prob',np.arange(1,52,50))
# Section1.Reliability.Overflow.Input = MechanismInput('Overflow')
# Section1.Reliability.Overflow.Input.fill_prob_mechanism(inputfile1,'OverflowInput')
#
# # INTEGRATED PROBABILISTIC
# Section1.Reliability.Overflow.generateLCRProfile(Section1.Reliability.Load,mechanism='Overflow',method='FORM',type='Prob',strength_params=Section1.Reliability.Overflow.Input)
# #FRAGILITY CURVES
# Section1.Reliability.Overflow.constructFragilityCurves(Section1.Reliability.Overflow.Input,start=5,step=0.2)
# Section1.Reliability.Overflow.generateLCRProfile(Section1.Reliability.Load,mechanism='Overflow',method='FORM')
# Section1.Reliability.Overflow.parameterizeLCR()


#
# Section1.Reliability.StabilityInner = MechanismReliabilityCollection('StabilityInner','Simple',np.arange(1,52,10))
# Section1.Reliability.StabilityInner.Input = MechanismInput('StabilityInner')
# Section1.Reliability.StabilityInner.Input.fill_prob_mechanism(inputfile1,'StabilityInnerInput')
# Section1.Reliability.StabilityInner.generateLCRProfile(Section1.Reliability.Load,mechanism='StabilityInner',type='Simple',strength_params=Section1.Reliability.StabilityInner.Input)
# #
# print('Analysis for piping for section 1')
Section1.Reliability.Piping = MechanismReliabilityCollection('Piping','Prob',np.arange(1,52,50))
Section1.Reliability.Piping.Input = MechanismInput('Piping')
Section1.Reliability.Piping.Input.fill_prob_mechanism(inputfile1,'PipingInput')
# print('Derive the fragility curve')
#PROBABILISTIC
Section1.Reliability.Piping.generateLCRProfile(Section1.Reliability.Load,mechanism='Piping',method='FORM',type='Prob',strength_params=Section1.Reliability.Piping.Input)
Section1b = copy.deepcopy(Section1)
#FC BASED
Section1.Reliability.Piping.constructFragilityCurves(Section1.Reliability.Piping.Input,start=5,step=0.2)
Section1.Reliability.Piping.generateLCRProfile(Section1.Reliability.Load,mechanism='Piping',method='FORM')

# Section1a.Reliability.Overflow.drawLCR(label='MCS integr')
Section1b.Reliability.Overflow.drawLCR(label='FORM integr')
pass
Section1.Reliability.Overflow.drawLCR(label='FORM FC')
plt.legend()
plt.show()

# # Section1.Reliability.Piping.parameterizeLCR()
# # Section1.Reliability.Piping.drawFC()
#
# # Section1.Reliability.Piping.drawLCR(type='beta')
# # Section1.Reliability.Overflow.drawLCR(type='beta')
#
# # plt.legend(('Piping','Overflow'))
# # plt.ylabel('beta')
# # plt.xlabel('tijd')
# # plt.show()
# #DONE: For each mechanism
# #DONE: We calculate current and future rate of failure without measures
#
# Strategy1 = Strategy('optimizationpersection')
# Strategy1.Definition.readinput(inputfile1,'Measures')
# Strategy1.InitialSituation = InitialSituation(Section1.Reliability)
# #We read a set of measures that are possible
# #We calculate current and future failure rates with these measures/strategies
# Strategy1.Result = StrategyCalculation(Strategy1.Definition,Strategy1.InitialSituation)
# #We combine for each set of (no) measures to a conditional failure rate on a section level. Therefore we
# #Upscale for length effects in the section
# #Translate to conditional failure rates
# #Combine the mechanisms
# Strategy1.Result.Measures['Soil reinforcement']['Overflow'].deriveBetaRelations(mechanism='Overflow',measure_type='Soil reinforcement',period=50)
# Strategy1.Result.Measures['Soil reinforcement']['StabilityInner'].deriveBetaRelations(mechanism='StabilityInner',measure_type='Soil reinforcement',period=50)
# Strategy1.Result.Measures['Soil reinforcement']['StabilityInner'].plotMeasureReliabilityinTime()
# Strategy1.Result.Measures['Soil reinforcement']['Overflow'].plotMeasureReliabilityinTime()
#
print(time.time()-start)
#
#
