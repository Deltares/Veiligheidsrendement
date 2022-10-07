import shelve
from pathlib import Path
from StrategyEvaluation import OverflowBundling
import copy

dir = r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\Documents\04_SAFE\SAFE repository\data\cases\SAFE_v0.6-16-4_testcombine\Case_first_run_20210326'
my_shelf = shelve.open(str(Path(dir).joinpath('TestOverflowBundling.out')))

Strategy = my_shelf['Strategy']
init_overflow_risk = my_shelf['init_overflow_risk']
existing_investment = my_shelf['existing_investment']
traject = my_shelf['traject']
# LifeCycleCost = my_shelf['LifeCycleCost']
LifeCycleCost = copy.deepcopy(Strategy.LCCOption)
section_order = [i.name for i in traject.Sections]


## No measures: WORKS!

# print('TEST FOR NO MEASURES:') #
# existing_investment = []
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment,
#                      LifeCycleCost,traject)
# print(measure_index)
# print(BC_out)
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#     print()
## TODO: check if answer is correct.

##Stability Screen: WORKS!
# print('\n \n TEST FOR STABILITY SCREEN:')
# print('Initial:')
# existing_investment = [(0,0,15)]
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
#
# print('Adapted:')
# for count, i in enumerate(measure_index[0]):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)


# ## Soil with SS
# print('\n \n TEST FOR STABILITY SCREEN WITH BERM (t=20): (Works)')
# print('Initial:')
# existing_investment = [(0,7,8)]
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)

## Soil with SS
# print('\n \n TEST FOR STABILITY SCREEN WITH BERM (t=0): WORKS')
# print('Initial:')
# existing_investment = [(0,10,10)]
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#
## Custom
# existing_investment = [(1,22,28),(0,14,16)]
# print('\n \n TEST FOR CUSTOM: (WORKS 99% sure)')
# print('Initial:')
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
# Strategy.get_measure_from_index(existing_investment[1],section_order,print_measure=True)
# #
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#
## Diaphragm Wall
# print('\n \n TEST FOR DIAPHRAGM WALL: Works')
# print('Initial:')
# existing_investment = [(0,13,13)]
# init_overflow_risk[0,:] = Strategy.RiskOverflow[0,13,:]
# # existing_investment = [(2,21,26)]
#
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
#
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)

## Soil reinforcement
# print('\n \n TEST FOR Soil reinforcement: WORKS')
# print('Initial:')
# existing_investment = [(0,4,5)]
# # existing_investment = [(2,21,26)]
#
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
#
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)

# ## Soil reinforcement t=20
# print('\n \n TEST FOR Soil reinforcement at t= 20: WORKS')
# print('Initial:')
# existing_investment = [(0,1,2)]
#
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
#
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#
# ## Vertical Geotextile
# print('\n \n TEST FOR Vertical Geotextile: WORKS')
# print('Initial:')
# existing_investment = [(0,0,13)]
# # existing_investment = [(2,21,26)]
#
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
# #
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)

## Soil t=20 + VZG
# print('\n \n TEST FOR Soil reinforcement at t= 20 with VZG: WORKS')
# print('Initial:')
# existing_investment = [(0,1,19)]
# # init_overflow_risk[0,:] = Strategy.RiskOverflow[0,13,:]
# # existing_investment = [(2,21,26)]
#
# Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)
#
# measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)
#
# print('Adapted:')
# for count, i in enumerate(measure_index):
#     Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#
#
# ## Soil + VZG
print('\n \n TEST FOR Soil reinforcement with VZG:')
print('Initial:')
existing_investment = [(0,4,23)]
# existing_investment = [(2,21,26)]

Strategy.get_measure_from_index(existing_investment[0],section_order,print_measure=True)

measure_index, BC_out = OverflowBundling(Strategy, init_overflow_risk,existing_investment, LifeCycleCost,traject)

print('Adapted:')
for count, i in enumerate(measure_index):
    Strategy.get_measure_from_index((count, measure_index[count][0], measure_index[count][1]),section_order,print_measure=True)
#
#


#OLD CODE
# set unusable measures to 999
#
#
# if full
# if not Soil reinforcement or Stability screen
# no changes possible
# else
# TODO THIS PART IS WRONG!
# if there is no investment sg yet
#
# possibleIndices = np.array([0])
# for j in np.argwhere(GeotechnicalOptions[('Section', 100)].values == GeotechnicalOptions.iloc[0][('Section', 100)]):        #dit is niet goed.
#     possibleIndices = np.append(possibleIndices, j + 1)
#
# if existing_investments[i,1] in possibleIndices:
#     #take the minimal LCC over all options, and corresponding sg index:
#     LCCs = np.min(LifeCycleCost[i, :, :], axis=1)
#     sg_indices[i,:] = np.argmin(LifeCycleCost[i, :, :], axis=1)
#
#     elif the year of the existing sg measure is >0 we will also consider situations where we pull this
#     investment forward to year 0, and then take the minimum of those measures (note that this is a slight
#     shortcut, not perfectly elegant) It works though.
# elif year_of_investment > 0:
#     #find indices with same berm width
#     currentberm = GeotechnicalOptions.iloc[existing_investments[i,1] - 1]['dberm'].values[0]
#     if GeotechnicalOptions['class'].iloc[existing_investments[i,1] - 1] != 'combined':
#         #indices of the same type:
#         indices = np.argwhere((GeotechnicalOptions['dberm'].values == currentberm) &(GeotechnicalOptions['type'].values == GeotechnicalOptions['type'].iloc[
#             existing_investments[i,1] - 1]) & (GeotechnicalOptions['year'].values == year_of_investment))
#         # indices = np.argwhere((GeotechnicalOptions['dberm'] == currentberm) &(GeotechnicalOptions['type'] == GeotechnicalOptions['type'].iloc[
#         #     existing_investments[i,1] - 1]))
#         #if it is a Geotextile or Stability Screen, it should also be possible to combine it with the corresponding soil reinforcement
#         if GeotechnicalOptions['type'].iloc[existing_investments[i,1] - 1] == 'Stability Screen':
#             stabilities50 = GeotechnicalOptions[('StabilityInner', 50)].iloc[indices.flatten()].values
#             indices = list(indices)
#             for stab in stabilities50:
#                 # indices.append(list(np.argwhere((GeotechnicalOptions['class'] == 'full') & (GeotechnicalOptions[('StabilityInner', 50)] == stab))))
#                 indices.append(list(np.argwhere((GeotechnicalOptions['class'] == 'full') & (GeotechnicalOptions[('StabilityInner', 50)] == stab) &
#                                                 (GeotechnicalOptions['year'] == year_of_investment))))
#             indice = [item for sublist in indices for item in sublist]
#             indices = np.unique(np.array(indice).flatten()).astype(np.int32)+1
#             pass
#         elif GeotechnicalOptions['type'].iloc[existing_investments[i,1] - 1] == 'Vertical Geotextile':
#             piping50 = GeotechnicalOptions[('Piping', 50)].iloc[indices.flatten()].values
#             indices = list(indices)
#             for pip in piping50:
#                 id1 = GeotechnicalOptions['ID'].iloc[existing_investments[i, 1] - 1]
#                 id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
#                 indices.append(list(np.argwhere(np.array(id_list) == id1)))
#             indice = [item for sublist in indices for item in sublist]
#             indices = np.unique(np.array(indice).flatten()).astype(np.int32)+1
#         elif GeotechnicalOptions['type'].iloc[existing_investments[i,1] - 1] == 'Custom':
#             pass
#         #this is incorrect!
#     else:
#         #find the indices where the first combined measure has the same ID
#         id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
#         ids = np.argwhere((np.array(id_list) == GeotechnicalOptions['ID'][existing_investments[i, 1] - 1][0])
#                         & (GeotechnicalOptions['dberm'].values == currentberm))
#         indices = np.add(ids.reshape((len(ids),)), 1)
#
#     #get costs and sg indices
#     LCC1 = LifeCycleCost[i, :, :]
#     LCC2 = LCC1[:, indices.flatten()]
#     LCCs = np.min(LCC2, axis=1)
#     sg_indices[i,:] = indices[np.argmin(LCC2, axis=1)].ravel()
#
# #else we don't want to change the geotechnical measure that is taken, so we only grab LCCs from that specific
# # column in the [n,sh,sg] LifeCycleCost array
# else:
#     if existing_investments[i,0] !=0:
#         LCCs = LifeCycleCost[i, existing_investments[i,0]:, existing_investments[i,1]]
#         # TCs = np.add(LCCs, np.sum(Strategy.RiskOverflow[i, existing_investments[i,0]:, :], axis=1))
#         sg_indices[i,:].fill(existing_investments[i,1])
#         index_existing = existing_investments[i,0]
#         if GeotechnicalOptions['type'][existing_investments[i, 1] - 1] == 'Custom':
#             pass
#     elif GeotechnicalOptions['type'][existing_investments[i, 1] - 1] == 'Vertical Geotextile': #it is a geotextile which should be combined with soil
#         id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
#         ids = np.argwhere(np.array(id_list) == GeotechnicalOptions['ID'][existing_investments[i, 1] - 1])
#         ids = np.add(ids.reshape((len(ids),)),1)
#         testLCC = LifeCycleCost[i, existing_investments[i, 0]:, ids].T
#         LCCs = np.min(testLCC,axis=1)
#         sg_indices[i,:] = np.array(ids)[np.argmin(testLCC, axis=1)]
#     elif GeotechnicalOptions['type'][existing_investments[i, 1] - 1] == 'Stability Screen':
#         ID_ref = GeotechnicalOptions['ID'][existing_investments[i, 1] - 1]
#         inv_year = GeotechnicalOptions.loc[GeotechnicalOptions['ID'] == ID_ref]['year'].values[0]
#         beta_investment_year = GeotechnicalOptions['StabilityInner',inv_year].loc[GeotechnicalOptions['ID'] == ID_ref].values[0]
#         #TODO Hier gaat het mis als je geen "full" opties hebt. Bijvoorbeeld als een scherm wel kan, maar je mag geen scherm icm met een kruinverhoging doen
#         ID_allowed = GeotechnicalOptions.loc[GeotechnicalOptions['StabilityInner',inv_year]==beta_investment_year]\
#             .loc[GeotechnicalOptions['type']=='Soil reinforcement'].loc[GeotechnicalOptions['class']=='full']['ID'].values[0]
#         ids = np.argwhere(GeotechnicalOptions['ID'].values == ID_allowed)
#         ids = np.add(ids.reshape((len(ids),)),1)
#         testLCC = LifeCycleCost[i, existing_investments[i, 0]:, ids].T
#         LCCs = np.min(testLCC,axis=1)
#         sg_indices[i,:] = np.array(ids)[np.argmin(testLCC, axis=1)]
#         #TODO Make a fix for stability screen: note this is not a very nice fix
#     else:
#         pass
#         #TODO Allow no changes!
#         #find the ID of stability screen
#
#         #it has to be combined with a soil reinforcement that is full, and where the betas for stability are identical
#
#         #find all geotechnical measures where the ID starts with the same number
#         #grab LCC values of these lines and take the minimum value, then find which sg index we should change to.
#
# at last we do some index manipulation:
# -to add the existing measure if we already have a geotechnical measure
# -to make options with cost 1e99 invalid
# sorted_indices[i, 0:len(LCCs)] = np.argsort(LCCs)+index_existing
# sorted_indices[i, 0:len(LCCs)] = np.where(np.sort(LCCs) > 1e60, 999, sorted_indices[i, 0:len(LCCs)])
# sg_indices[i, 0:len(LCCs)] = sg_indices[i,0:len(LCCs)][np.argsort(LCCs)]