'''This is a file with all the general configuration settings for the SAFE computations
Use them in a function by calling import config, and then config.key'''

from pathlib import Path
import pandas as pd
import sys
import os
import subprocess
import shutil

## GENERAL SETTINGS
timing = False
path = Path(sys.path[0])
traject = '16-3'

#manage the in and output directory of the test (this should be done somewhere else eventually)
git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
casename = '{}'.format(git_hash)

directory = path.joinpath(casename)
if not directory.exists():
    directory.mkdir(exist_ok=True)
else:
    shutil.rmtree(directory)
    directory.mkdir(exist_ok=False)

directory.joinpath('figures').mkdir(parents=True)
directory.joinpath('results', 'investment_steps').mkdir(parents=True)

language = 'EN'

## OUTPUT SETTINGS:
#General settings:
shelves = False                                              #setting to shelve intermediate results
reuse_output = False                                        #reuse intermediate result if available
beta_or_prob = 'beta'                                       #whether to use 'beta' or 'prob' for plotting reliability

#Settings for step 1:
plot_reliability_in_time = False                              #Setting to turn on plotting the reliability in time for each section.
plot_measure_reliability = False                              #Setting to turn on plotting beta of measures at each section.
flip_traject = True                                         #Setting to flip the direction of the longitudinal plots. Used for SAFE as sections are numbered east-west
assessment_plot_years = [0,20,50]                           #years (relative to t_0) to plot the reliability


#Settings for step 2:
geometry_plot = False                                       #Setting to plot the change in geometry for each soil reinforcement combination. Only use for debugging: very time consuming.

#Settings for step 3:
design_methods = ['Veiligheidsrendement','Doorsnede-eisen'] #Design methods (do not change, if you do ensure that the proper keys are used)

#dictionary with settings for beta-cost curve:
beta_cost_settings = {'symbols':True,                       #whether to include symbols in the beta-cost curve
                      'markersize':10}                      #base size of markers.
#unit costs:
unit_cost_data = pd.read_csv('../../tools/unit_costs.csv', encoding='latin_1')


unit_cost = {}
for count, i in unit_cost_data.iterrows():
    unit_cost[i['Description']] = i['Cost']


#TODO add all plot & language settings here.