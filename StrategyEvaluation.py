import pandas as pd
import numpy as np
import copy
import ProbabilisticFunctions
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import config
# This script combines two sets of measures to a single option
def MeasureCombinations(combinables, partials, solutions,splitparams = False):
    CombinedMeasures = pd.DataFrame(columns=combinables.columns)

    #loop over partials
    for i, row1 in partials.iterrows():
    #combine with all combinables
        for j, row2 in combinables.iterrows():

            ID = '+'.join((row1['ID'].values[0], row2['ID'].values[0]))
            types = [row1['type'].values[0], row2['type'].values[0]]
            year = [row1['year'].values[0], row2['year'].values[0]]
            if splitparams:
                params = [row1['yes/no'].values[0], row2['dcrest'].values[0],row2['dberm'].values[0]]
            else:
                params = [row1['params'].values[0], row2['params'].values[0]]
            Cost = [row1['cost'].values[0], row2['cost'].values[0]]
            #combine betas
            #take maximums of mechanisms except if it is about StabilityInner for partial Stability Screen
            betas = []
            years = []

            for ij in partials.columns:
                if ij[0] != 'Section' and ij[1] != '':     #It is a beta value
                    beta = np.maximum(row1[ij], row2[ij])
                    years.append(ij[1])
                    betas.append(beta)

            #next update section probabilities
            for ij in partials.columns:
                if ij[0] =='Section':     #It is a beta value
                    #where year in years is the same as ij[1]
                    indices = [indices for indices, x in enumerate(years) if x == ij[1]]
                    ps = ProbabilisticFunctions.beta_to_pf(np.array(betas)[indices])
                    p = np.sum(ps)
                    betas.append(ProbabilisticFunctions.pf_to_beta(p))
                    # print(ProbabilisticFunctions.pf_to_beta(p)-np.max([row1[ij],row2[ij]]))
                    # if ProbabilisticFunctions.pf_to_beta(p)-np.max([row1[ij],row2[ij]]) > 1e-8:
                    #     pass
            if splitparams:
                in1 = [ID, types, 'combined', year, params[0],params[1],params[2], Cost]
            else:
                in1 = [ID, types, 'combined', year, params, Cost]

            allin = pd.DataFrame([in1 + betas], columns=combinables.columns)
            CombinedMeasures = CombinedMeasures.append(allin)
    return CombinedMeasures
def getTrajectProb(traject, traject_prob,trange):
    for mechanism in range(0,len(traject.GeneralInfo['Mechanisms'])):
        traject_prob[mechanism, :, :] = traject.Probabilities.loc[
            traject.Probabilities['index'] == traject.GeneralInfo['Mechanisms'][mechanism]].drop(
            ['index', 'Section', 'Length'], axis=1).values
    return traject_prob
def makeTrajectDF(traject, cols):
    # cols = cols[1:]
    sections = []

    for i in traject.Sections:
        sections.append(i.name)

    mechanisms = list(traject.Sections[0].MechanismData.keys()) + ['Section']
    df_index = pd.MultiIndex.from_product([sections, mechanisms], names=['name', 'mechanism'])
    TrajectProbability = pd.DataFrame(columns=cols, index=df_index)

    for i in traject.Sections:
        for j in mechanisms:
            TrajectProbability.loc[(i.name, j)] = list(i.Reliability.SectionReliability.loc[j])

    return TrajectProbability

#hereafter a bunch of functions to compute costs, risks and probabilities over time are defined:
def calcTC(section_options, r=0.03, horizon=100):
    costs = section_options['cost'].values
    years = section_options['year'].values
    discountfactors = list(map(lambda x: 1 / (1 + r) ** np.array(x), years))
    TC = list(map(lambda c, r: c * r, costs, discountfactors))
    return np.array(list(map(lambda c: np.sum(c), TC)))

def calcTR(section, section_options, base_traject, original_section, r=0.03, horizon=100, damage=1e9):
    #section: the section name
    #section_options: all options for the section
    #base_traject: traject probability with all implemented measures
    #takenmeasures: object with all measures taken
    #original section: series of probabilities of section, before taking a measure.
    if damage == 1e9:
        print('WARNING NO DAMAGE DEFINED')

    TotalRisk = []
    dR = []
    mechs = np.unique(base_traject.index.get_level_values('mechanism').values)
    sections = np.unique(base_traject.index.get_level_values('name').values)
    section_idx = np.where(sections == section)[0]
    section_options_array = {}
    base_array = {}
    TotalRisk = []
    dR = []

    for i in mechs:
        base_array[i] = base_traject.xs(i, level=1).values.astype('float')
        if isinstance(section_options, pd.DataFrame):
            section_options_array[i] = section_options.xs(i, level=0, axis=1).values.astype('float')
            range_idx = len(section_options_array[mechs[0]])

        if isinstance(section_options, pd.Series):
            section_options_array[i] = section_options.xs(i, level=0).values.astype('float')
            range_idx = 0

    if 'section_options_array' in locals():
        base_risk = calcLifeCycleRisks(base_array, r, horizon, damage, datatype='Array', ts=base_traject.columns.values, mechs=mechs)

        for i in range(range_idx):
                TR = calcLifeCycleRisks(base_array, r, horizon, damage, change=section_options_array, section=section_idx, datatype='Array', ts=base_traject.columns.values, mechs=mechs, option=i)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)
    else:
        base_risk = calcLifeCycleRisks(base_traject, r, horizon, damage)
        if isinstance(section_options, pd.DataFrame):
            for i, row in section_options.iterrows():
                TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change=row, section=section)
                TotalRisk.append(TR)
                dR.append(base_risk - TR)

        elif isinstance(section_options, pd.Series):
            TR = calcLifeCycleRisks(base_traject, r, horizon, damage, change=section_options, section=section)
            TotalRisk.append(TR)
            dR.append(base_risk - TR)

    return base_risk, dR, TotalRisk

def calcLifeCycleRisks(base0, r, horizon,damage, change=None, section=None, datatype='DataFrame', ts=None,mechs=False,
                       option=None,dumpPt=False):
    base = copy.deepcopy(base0)
    if datatype == 'DataFrame':
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        if isinstance(change, pd.Series):
            for i in mechs:
                #This is not very efficient. Could be improved.
                base.loc[(section, i)] = change.loc[i]
        else:
            pass

        beta_t, p_t = calcTrajectProb(base, horizon=horizon)
    elif datatype == 'Array':
        if isinstance(change, dict):
            for i in mechs:
                base[i][section] = change[i][option]
        else:
            pass
        if not (isinstance(ts,np.ndarray) or isinstance(ts,list)):
            ts = np.array(range(0,horizon))
        if not isinstance(mechs,np.ndarray): mechs = np.array(list(base.keys()))
        beta_t, p_t = calcTrajectProb(base, horizon=horizon, datatype='Arrays', ts=ts, mechs=mechs)

    # trange = np.arange(0, horizon + 1, 1)
    trange = np.arange(0, horizon, 1)
    D_t = damage / (1 + r) ** trange
    risk_t = p_t * D_t
    if dumpPt:
        np.savetxt(dumpPt,p_t,delimiter=",")
    TR = np.sum(risk_t)
    return TR

def calcTrajectProb(base, horizon=False, datatype='DataFrame', ts=None, mechs=False):
    pfs = {}
    if horizon:
        trange = np.arange(0, horizon, 1)
    elif ts:
        trange = [ts]
    else:
        raise ValueError('No range defined')
    if datatype == 'DataFrame':
        ts = base.columns.values
        if not mechs:
            mechs = np.unique(base.index.get_level_values('mechanism').values)
        # mechs = ['Overflow']
    # pf_traject = np.zeros((len(ts),))
    pf_traject = np.zeros((len(trange),))

    for i in mechs:
        if i != 'Section':
            if datatype == 'DataFrame':
                betas = base.xs(i, level='mechanism').values.astype('float')
            else:
                betas = base[i]
            beta_interp = interp1d(np.array(ts).astype(np.int),betas)
            pfs[i] = ProbabilisticFunctions.beta_to_pf(beta_interp(trange))
            # pfs[i] = ProbabilisticFunctions.beta_to_pf(betas)
            pnonfs = 1 - pfs[i]
            if i == 'Overflow':
                # pf_traject += np.max(pfs[i], axis=0)
                pf_traject  = 1-np.multiply(1-pf_traject,1-np.max(pfs[i], axis=0))
            else:
                # pf_traject += np.sum(pfs[i], axis=0)
                # pf_traject += 1-np.prod(pnonfs, axis=0)
                pf_traject  = 1-np.multiply(1-pf_traject,np.prod(pnonfs, axis=0))

    ## INTERPOLATION AFTER COMBINATION:
    # pfail = interp1d(ts,pf_traject)
    # p_t1 = ProbabilisticFunctions.beta_to_pf(pfail(trange))
    # betafail = interp1d(ts, ProbabilisticFunctions.pf_to_beta(pf_traject),kind='linear')
    # beta_t = betafail(trange)
    # p_t = ProbabilisticFunctions.beta_to_pf(np.array(beta_t, dtype=np.float64))

    beta_t = ProbabilisticFunctions.pf_to_beta(pf_traject)
    p_t = pf_traject
    return beta_t, p_t

#this function changes the trajectprobability of a measure is implemented:
def ImplementOption(section, TrajectProbability, newProbability):
    mechs = np.unique(TrajectProbability.index.get_level_values('mechanism').values)
    #change trajectprobability by changing probability for each mechanism
    for i in mechs:
        TrajectProbability.loc[(section, i)] = newProbability[i]
    return TrajectProbability

def split_options(options):
    options_height = copy.deepcopy(options)
    options_geotechnical = copy.deepcopy(options)
    for i in options:
        #filter all different measures for height
        options_height[i] = options_height[i].loc[options[i]['class'] != 'combined']
        options_height[i] = options_height[i].loc[(options[i]['type'] == 'Diaphragm Wall') | (options[i]['dberm'] == 0)]




        #now we filter all geotechnical measures
        #first all crest heights are thrown out
        options_geotechnical[i] = options_geotechnical[i].loc[
            (options_geotechnical[i]['dcrest'] == 0.0) | (options_geotechnical[i]['dcrest']==-999) |
            ((options_geotechnical[i]['class'] == 'combined') & (options_geotechnical[i]['dberm'] == 0))]

        #subtract startcosts, only for height.
        startcosts = np.min(options_height[i][(options_height[i]['type'] == 'Soil reinforcement')]['cost'])
        options_height[i]['cost'] = np.where(options_height[i]['type'] == 'Soil reinforcement',
                                             np.subtract(options_height[i]['cost'], startcosts),
                                             options_height[i]['cost'])

        # if an option has a stability screen, the costs for height are too high. This has to be adjusted. We do this
        # for all soil reinforcements. costs are not discounted yet, so we can disregard the year of the investment:
        for ij in np.unique(options_height[i].loc[options_height[i]['type']=='Soil reinforcement']['dcrest']):
            options_height[i].loc[options_height[i]['dcrest'] == ij,'cost'] = np.min(options_height[i].loc[options_height[i]['dcrest'] == ij]['cost'])

        options_geotechnical[i] = options_geotechnical[i].reset_index(drop=True)
        options_height[i]       = options_height[i].reset_index(drop=True)

        #loop for the geotechnical stuff:
        newcosts = []
        for ij in options_geotechnical[i].index:
            if options_geotechnical[i].iloc[ij]['type'].values[0] == 'Soil reinforcement':
                newcosts.append(options_geotechnical[i].iloc[ij]['cost'].values[0])
            elif options_geotechnical[i].iloc[ij]['class'].values[0] == 'combined':
                newcosts.append([options_geotechnical[i].iloc[ij]['cost'].values[0][0],
                                 options_geotechnical[i].iloc[ij]['cost'].values[0][1]])
            else:
                newcosts.append(options_geotechnical[i].iloc[ij]['cost'].values[0])
        options_geotechnical[i]['cost'] = newcosts
        #only keep reliability of relevant mechanisms in dictionary
        options_height[i].drop(['Piping','StabilityInner','Section'],axis=1,level=0)
        options_geotechnical[i].drop(['Overflow','Section'],axis=1,level=0)
    return options_height,options_geotechnical

def SolveMIP(MIPModel):

    MixedIntegerSolution = MIPModel.solve()
    return MixedIntegerSolution

def evaluateRisk(init_overflow_risk,init_geo_risk,Strategy,n,sh,sg):
    for i in config.mechanisms:
        if i == 'Overflow':
            init_overflow_risk[n,:] = Strategy.RiskOverflow[n,sh,:]
        else:
            init_geo_risk[n,:] = Strategy.RiskGeotechnical[n,sg,:]
    return init_overflow_risk,init_geo_risk

def updateProbability(init_probability,Strategy, index):
    '''index = [n,sh,sg]'''
    for i in init_probability:
        from scipy.stats import norm
        # plt.plot(-norm.ppf(init_probability[i][index[0],:]), 'r')
        if i == 'Overflow':
            init_probability[i][index[0],:] = Strategy.Pf[i][index[0],index[1],:]
        else:
            init_probability[i][index[0], :] = Strategy.Pf[i][index[0], index[2], :]
        # plt.plot(-norm.ppf(init_probability[i][index[0],:]),'b')
        # plt.savefig('Beta ' + i + str(index) + '.png')
        # plt.close()
    return init_probability

def OverflowBundling(Strategy, init_overflow_risk, BCref,existing_investment,
                     LifeCycleCost,traject):
    '''Routine for bundling several measures for overflow to prevent getting stuck if many overflow-dominated
    sections have about equal reliability. A bundle is a set of measures (typically crest heightenings) at different sections.
    This routine is needed for mechanisms where the system reliability is computed as a series system with fully correlated components.'''

    #Step 1: fill an array of size (n,2) with sh and sg of existing investments per section in order to properly filter
    # the viable options per section
    existing_investments = np.zeros((np.size(LifeCycleCost, axis=0), 2), dtype=np.int32)
    if len(existing_investment) > 0:
        for i in range(0,len(existing_investment)):
            existing_investments[existing_investment[i][0],0] = existing_investment[i][1]    #sh
            existing_investments[existing_investment[i][0],1] = existing_investment[i][2]    #sg

    #Step 2: for each section, determine the sorted_indices of the min to max LCC. Note that this could also be based on TC but the performance is good as is.
    #first make the proper arrays for sorted_indices (sh), corresponding sg indices and the LCC for each section.
    sorted_indices = np.empty((np.size(LifeCycleCost, axis=0), np.size(LifeCycleCost, axis=1)+1),dtype=np.int32)
    sorted_indices.fill(999)
    LCC_values = np.zeros((np.size(LifeCycleCost, axis=0),))
    sg_indices = np.empty((np.size(LifeCycleCost, axis=0), np.size(LifeCycleCost, axis=1)),dtype=np.int32)
    sg_indices.fill(999)

    #loop over the sections
    for i in range(0, len(traject.Sections)):
        index_existing = 0  #value is only used in 1 of the branches of the if statement, otherwise should be 0.
        #get the indices where safety is equal to no measure for stabilityinner & piping
        #if there are investments this loop is needed to deal with the fact that it can be an integer or list.
        if existing_investments[i,1] != 0:
            if isinstance(Strategy.options_geotechnical[traject.Sections[i].name].iloc[existing_investments[i,1]-1]['year'].values[0],list):
                year_of_investment = Strategy.options_geotechnical[traject.Sections[i].name].iloc[existing_investments[i,1]-1]['year'].values[0][-1]
            elif isinstance(Strategy.options_geotechnical[traject.Sections[i].name].iloc[existing_investments[i,1]-1]['year'].values[0],int):
                year_of_investment = Strategy.options_geotechnical[traject.Sections[i].name].iloc[existing_investments[i,1]-1]['year'].values[0]

        #main routine:
        GeotechnicalOptions = Strategy.options_geotechnical[traject.Sections[i].name]

        # if there is no investment sg yet
        possibleIndices = np.array([0])
        for j in np.argwhere(GeotechnicalOptions[('Section', 100)].values == GeotechnicalOptions.iloc[0][('Section', 100)]):
            possibleIndices = np.append(possibleIndices, j + 1)
        if existing_investments[i,1] in possibleIndices:
            #take the minimal LCC over all options, and corresponding sg index:
            LCCs = np.min(LifeCycleCost[i, :, :], axis=1)
            sg_indices[i,:] = np.argmin(LifeCycleCost[i, :, :], axis=1)

        #elif the year of the existing sg measure is >0 we will also consider situations where we pull this
        # investment forward to year 0, and then take the minimum of those measures (note that this is a slight
        # shortcut, not perfectly elegant) It works though.
        elif year_of_investment > 0:
            #find indices with same berm width
            currentberm = GeotechnicalOptions.iloc[existing_investments[i,1] - 1]['dberm'].values[0]
            if GeotechnicalOptions['class'].iloc[existing_investments[i,1] - 1] != 'combined':
                #indices of the same type:
                indices = np.argwhere((GeotechnicalOptions['dberm'] == currentberm) &(GeotechnicalOptions['type'] == GeotechnicalOptions['type'].iloc[
                    existing_investments[i,1] - 1]) & (GeotechnicalOptions['year'] == year_of_investment))
                # indices = np.argwhere((GeotechnicalOptions['dberm'] == currentberm) &(GeotechnicalOptions['type'] == GeotechnicalOptions['type'].iloc[
                #     existing_investments[i,1] - 1]))
                #if it is a Geotextile or Stability Screen, it should also be possible to combine it with the corresponding soil reinforcement
                if GeotechnicalOptions['type'].iloc[existing_investments[i,1] - 1] == 'Stability Screen':
                    stabilities50 = GeotechnicalOptions[('StabilityInner', 50)].iloc[indices.flatten()].values
                    indices = list(indices)
                    for stab in stabilities50:
                        # indices.append(list(np.argwhere((GeotechnicalOptions['class'] == 'full') & (GeotechnicalOptions[('StabilityInner', 50)] == stab))))
                        indices.append(list(np.argwhere((GeotechnicalOptions['class'] == 'full') & (GeotechnicalOptions[('StabilityInner', 50)] == stab) &
                                                        (GeotechnicalOptions['year'] == year_of_investment))))
                    indice = [item for sublist in indices for item in sublist]
                    indices = np.unique(np.array(indice).flatten()).astype(np.int32)+1
                    pass
                elif GeotechnicalOptions['type'].iloc[existing_investments[i,1] - 1] == 'Vertical Geotextile':
                    piping50 = GeotechnicalOptions[('Piping', 50)].iloc[indices.flatten()].values
                    indices = list(indices)
                    for pip in piping50:
                        id1 = GeotechnicalOptions['ID'].iloc[existing_investments[i, 1] - 1]
                        id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
                        indices.append(list(np.argwhere(np.array(id_list) == id1)))
                    indice = [item for sublist in indices for item in sublist]
                    indices = np.unique(np.array(indice).flatten()).astype(np.int32)+1
                #this is incorrect!
            else:
                #find the indices where the first combined measure has the same ID
                id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
                ids = np.argwhere((np.array(id_list) == GeotechnicalOptions['ID'][existing_investments[i, 1] - 1][0])
                                & (GeotechnicalOptions['dberm'].values == currentberm))
                indices = np.add(ids.reshape((len(ids),)), 1)

            #get costs and sg indices
            LCC1 = LifeCycleCost[i, :, :]
            LCC2 = LCC1[:, indices.flatten()]
            LCCs = np.min(LCC2, axis=1)
            sg_indices[i,:] = indices[np.argmin(LCC2, axis=1)].ravel()

        #else we don't want to change the geotechnical measure that is taken, so we only grab LCCs from that specific
        # column in the [n,sh,sg] LifeCycleCost array
        else:
            if existing_investments[i,0] !=0:
                LCCs = LifeCycleCost[i, existing_investments[i,0]:, existing_investments[i,1]]
                # TCs = np.add(LCCs, np.sum(Strategy.RiskOverflow[i, existing_investments[i,0]:, :], axis=1))
                sg_indices[i,:].fill(existing_investments[i,1])
                index_existing = existing_investments[i,0]
            elif GeotechnicalOptions['type'][existing_investments[i, 1] - 1] == 'Vertical Geotextile': #it is a geotextile which should be combined with soil
                id_list = [j[0] for j in GeotechnicalOptions['ID'].values]
                ids = np.argwhere(np.array(id_list) == GeotechnicalOptions['ID'][existing_investments[i, 1] - 1])
                ids = np.add(ids.reshape((len(ids),)),1)
                testLCC = LifeCycleCost[i, existing_investments[i, 0]:, ids].T
                LCCs = np.min(testLCC,axis=1)
                sg_indices[i,:] = np.array(ids)[np.argmin(testLCC, axis=1)]
            elif GeotechnicalOptions['type'][existing_investments[i, 1] - 1] == 'Stability Screen':
                ID_ref = GeotechnicalOptions['ID'][existing_investments[i, 1] - 1]
                inv_year = GeotechnicalOptions.loc[GeotechnicalOptions['ID'] == ID_ref]['year'].values[0]
                beta_investment_year = GeotechnicalOptions['StabilityInner',inv_year].loc[GeotechnicalOptions['ID'] == ID_ref].values[0]
                #TODO Hier gaat het mis als je geen "full" opties hebt. Bijvoorbeeld als een scherm wel kan, maar je mag geen scherm icm met een kruinverhoging doen
                ID_allowed = GeotechnicalOptions.loc[GeotechnicalOptions['StabilityInner',inv_year]==beta_investment_year]\
                    .loc[GeotechnicalOptions['type']=='Soil reinforcement'].loc[GeotechnicalOptions['class']=='full']['ID'].values[0]
                ids = np.argwhere(GeotechnicalOptions['ID'] == ID_allowed)
                ids = np.add(ids.reshape((len(ids),)),1)
                testLCC = LifeCycleCost[i, existing_investments[i, 0]:, ids].T
                LCCs = np.min(testLCC,axis=1)
                sg_indices[i,:] = np.array(ids)[np.argmin(testLCC, axis=1)]
                #TODO Make a fix for stability screen: note this is not a very nice fix
                #find the ID of stability screen

                #it has to be combined with a soil reinforcement that is full, and where the betas for stability are identical

                #find all geotechnical measures where the ID starts with the same number
                #grab LCC values of these lines and take the minimum value, then find which sg index we should change to.

        #at last we do some index manipulation:
        #-to add the existing measure if we already have a geotechnical measure
        #-to make options with cost 1e99 invalid
        sorted_indices[i, 0:len(LCCs)] = np.argsort(LCCs)+index_existing
        sorted_indices[i, 0:len(LCCs)] = np.where(np.sort(LCCs) > 1e60, 999, sorted_indices[i, 0:len(LCCs)])
        sg_indices[i, 0:len(LCCs)] = sg_indices[i,0:len(LCCs)][np.argsort(LCCs)]

    new_overflow_risk = copy.deepcopy(init_overflow_risk)
    #Step 3: determine various bundles for overflow:

    #first initialize som values
    index_counter = np.zeros((len(traject.Sections),),dtype=np.int32)      #counter that keeps track of the next cheapest option for each section
    run_number = 0                                                         #used for counting the loop
    counter_list = []                                                      #used to store the bundle indices
    BC_list = []                                                           #used to store BC for each bundle
    weak_list = []                                                         #used to store index of weakest section

    #here we start the loop. Note that we rarely make it to run 100, for larger problems this limit might need to be increased
    #TODO think about a more proper rule to exit the routine
    while run_number < 100:

        #get weakest section
        ind_weakest = np.argmax(np.sum(new_overflow_risk, axis=1))

        #take next step, exception if there is no valid measure. In that case exit the routine.
        if sorted_indices[ind_weakest, index_counter[ind_weakest]] == 999:
            # print('Bundle quit, weakest section has no more available measures')
            break

        #insert next cheapest measure from sorted list into overflow risk, then compute the LCC value and BC
        new_overflow_risk[ind_weakest, :] = Strategy.RiskOverflow[ind_weakest,
                                            sorted_indices[ind_weakest, index_counter[ind_weakest]], :]
        LCC_values[ind_weakest] = np.min(LifeCycleCost[ind_weakest,sorted_indices[ind_weakest,index_counter[ind_weakest]],
                                                                    sg_indices[ind_weakest,index_counter[ind_weakest]]])
        BC = (np.sum(np.max(init_overflow_risk,axis=0)) - np.sum(np.max(new_overflow_risk,axis=0))) / np.sum(
            LCC_values)

        #store results of step:
        BC_list.append(BC)
        weak_list.append(ind_weakest)

        #in the next step, the next measure should be taken for this section
        index_counter[ind_weakest] += 1

        #store the bundle indices, do -1 as index_counter contains the NEXT step
        counter_list.append(copy.deepcopy(index_counter)-1)

        run_number += 1

    #take the final index from the list, where BC is max
    if len(BC_list) > 0:
        final_index = counter_list[np.argmax(BC_list)]
        # convert measure_index to sh based on sorted_indices
        sg_index = np.empty((len(traject.Sections),))
        measure_index = np.zeros((np.size(LifeCycleCost, axis=0),), dtype=np.int32)
        for i in range(0, len(measure_index)):
            if final_index[i] != -1:  # a measure was taken
                measure_index[i] = sorted_indices[i, final_index[i]]
                sg_index[i] = sg_indices[i, final_index[i]]
            else:  # no measure was taken
                measure_index[i] = existing_investments[i, 0]
                sg_index[i] = existing_investments[i, 1]

        measure_index = np.append(measure_index, sg_index).reshape((2, len(traject.Sections))).T.astype(np.int32)
        BC_out = np.max(BC_list)
    else:
        BC_out = 0
        measure_index = []
        print('Warning: no more measures for weakest overflow section')



    return measure_index, BC_out















