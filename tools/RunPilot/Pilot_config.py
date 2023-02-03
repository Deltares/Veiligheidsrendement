'''This is a file with all the general configuration settings for the SAFE computations
Use them in a function by calling import config, and then config.key'''
import warnings
from pathlib import Path
import pandas as pd
import sys
import os
import subprocess
import shutil

## GENERAL SETTINGS
timing = True

traject = '38-1'
# path = Path(r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Prioritering WSRL\Gegevens 38-1\VR_berekening')
path = Path(r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Prioritering WSRL\Gegevens 38-1\VR_showcase')

casename = 'run_0'

directory = path.joinpath('Case_' + casename)
language = 'NL'

directory = path.joinpath(casename)
if not directory.exists():
    directory.joinpath('figures').mkdir(parents=True)
    directory.joinpath('results', 'investment_steps').mkdir(parents=True)
else:
    # shutil.rmtree(directory)
    # directory.mkdir(exist_ok=False)
    pass


## OUTPUT SETTINGS:
#General settings:
shelves = True                                              #setting to shelve intermediate results
reuse_output = True                                        #reuse intermediate result if available
beta_or_prob = 'beta'                                       #whether to use 'beta' or 'prob' for plotting reliability

#Settings for step 1:
plot_reliability_in_time = False                              #Setting to turn on plotting the reliability in time for each section.
plot_measure_reliability = False                              #Setting to turn on plotting beta of measures at each section.
flip_traject = True                                         #Setting to flip the direction of the longitudinal plots. Used for SAFE as sections are numbered east-west
assessment_plot_years = [0,20,50]                           #years (relative to t_0) to plot the reliability


#Settings for step 2:
geometry_plot = False                                       #Setting to plot the change in geometry for each soil reinforcement combination. Only use for debugging: very time consuming.

#Settings for step 3:
design_methods = ['Veiligheidsrendement','Doorsnede-eisen'] #Design methods (do not change, if you do, ensure that the proper keys are used)

#dictionary with settings for beta-cost curve:
beta_cost_settings = {'symbols':True,                       #whether to include symbols in the beta-cost curve
                      'markersize':10}                      #base size of markers.
#unit costs:
try:
    unit_cost_data = pd.read_csv('../unit_costs.csv', encoding='latin_1')
    unit_cost = {}
    for count, i in unit_cost_data.iterrows():
        unit_cost[i['Description']] = i['Cost']
except:
    pass
    warnings.warn('Warning: could not load unit costs!')

#TODO add all plot & language settings here.