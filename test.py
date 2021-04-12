'''This is a file with all the general configuration settings for the SAFE computations
Use them in a function by calling import config, and then config.key'''


# data opslaan: regel 973 en verder in Strategy.py
import numpy as np
import shelve
# import config

my_shelf = shelve.open(str(r'c:\Users\krame_n\0_WERK\SAFE\Repos\data\cases\Testcase_SAFE_v08_test_16-4\Case_adapted_overflow_20210330_2\FinalGreedyResult.out'))
Strategy = my_shelf['Strategy']
solutions = my_shelf['solutions']
measure_list = my_shelf['measure_list']
BC_list = my_shelf['BC_list']
Probabilities = my_shelf['Probabilities']

my_shelf2 = shelve.open(str(r'c:\Users\krame_n\0_WERK\SAFE\Repos\data\cases\Testcase_SAFE_v08_test_16-4\Case_adapted_overflow_20210330_2\AfterStep1.out'))
traject = my_shelf2['TrajectObject']
Strategy.writeGreedyResults(traject,solutions,measure_list,BC_list,Probabilities)
