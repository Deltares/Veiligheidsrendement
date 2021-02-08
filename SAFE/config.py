'''This is a file with all the general configuration settings for the SAFE computations
Use them in a function by calling import config, and then config.key'''
from pathlib import Path

## GENERAL SETTINGS
timing = True
traject = '16-4'
path = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\Testcase_fast_' + traject)
casename = 'reference_results'
directory = path.joinpath('Case_' + casename)
language = 'NL'

## RELIABILITY COMPUTATION
t_0 = 2025                                                  #year the computation starts
T = [0, 19, 20, 25, 50, 75, 100]                            #years to compute reliability for
mechanisms=['Overflow', 'StabilityInner','Piping']          #mechanisms to consider
LE_in_section=False                                         #whether to consider length-effects within a dike section

## OPTIMIZATION SETTINGS
OI_year = 0                                                 #investment year for TargetReliabilityBased approach
OI_horizon = 50                                             #Design horizon for TargetReliabilityBased approach
design_methods = ['Veiligheidsrendement','Doorsnede-eisen'] #Design methods (do not change, if you do ensure that the proper keys are used)
BC_stop = 0.1                                               #Stop criterion for benefit-cost ratio
max_greedy_iterations = 150                                 #maximum number of iterations in the greedy search algorithm
f_cautious = 1.5                                            #cautiousness factor for the greedy search algorithm. Larger values result in larger steps but lower accuracy and larger probability of finding a local optimum

## OUTPUT SETTINGS:
#General settings:
shelves = True                                              #setting to shelve intermediate results
reuse_output = False                                        #reuse intermediate result if available
beta_or_prob = 'beta'                                       #whether to use 'beta' or 'prob' for plotting reliability

#Settings for step 1:
plot_reliability_in_time=False                              #Setting to turn on plotting the reliability in time for each section.

flip_traject = True                                         #Setting to flip the direction of the longitudinal plots. Used for SAFE as sections are numbered east-west
assessment_plot_years = [0,20,50]                           #years (relative to t_0) to plot the reliability


#Settings for step 2:
geometry_plot = False                                       #Setting to plot the change in geometry for each soil reinforcement combination. Only use for debugging: very time consuming.

#Settings for step 3:

#dictionary with settings for beta-cost curve:
beta_cost_settings = {'symbols':True,                       #whether to include symbols in the beta-cost curve
                      'markersize':10}                      #base size of markers.


#TODO add all plot & language settings here.