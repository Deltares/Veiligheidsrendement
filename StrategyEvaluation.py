import pandas as pd
import numpy as np
import copy
from scipy.stats import norm
from scipy.interpolate import interp1d

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
                    ps = norm.cdf(-np.array(betas)[indices])
                    p = np.sum(ps)
                    betas.append(-norm.ppf(p))

            if splitparams:
                in1 = [ID, types, 'combined', year, params[0],params[1],params[2], Cost]
            else:
                in1 = [ID, types, 'combined', year, params, Cost]

            allin = pd.DataFrame([in1 + betas], columns=combinables.columns)
            CombinedMeasures = CombinedMeasures.append(allin)
    return CombinedMeasures

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

def calcTrajectProb(base, horizon=None, datatype='DataFrame', ts=None, mechs=False):
    pfs = {}
    trange = np.arange(0, horizon, 1)
    if datatype == 'DataFrame':
        ts = base.columns.values
        mechs = np.unique(base.index.get_level_values('mechanism').values)
        # mechs = ['Overflow']
    pf_traject = np.zeros((len(ts),))
    # pf_traject = np.zeros((len(trange),))

    for i in mechs:
        if i != 'Section':
            if datatype == 'DataFrame':
                betas = base.xs(i, level='mechanism').values.astype('float')
            else:
                betas = base[i]
            beta_interp = interp1d(ts,betas)
            pfs[i] = norm.cdf(-beta_interp(trange))
            # pfs[i] = norm.cdf(-betas)
            pnonfs = 1 - pfs[i]
            if i == 'Overflow':
                pf_traject += np.max(pfs[i], axis=0)
            else:
                pf_traject += np.sum(pfs[i], axis=0)
                # pf_traject += 1-np.prod(pnonfs, axis=0)

    ## INTERPOLATION AFTER COMBINATION:
    # pfail = interp1d(ts,pf_traject)
    # p_t1 = norm.cdf(-pfail(trange))
    # betafail = interp1d(ts, -norm.ppf(pf_traject),kind='linear')
    # beta_t = betafail(trange)
    # p_t = norm.cdf(-np.array(beta_t, dtype=np.float64))

    beta_t = -norm.ppf(pf_traject)
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
            options_height[i].loc[options_height[i]['dcrest'] == ij, 'cost'] = np.min(
                options_height[i].loc[options_height[i]['dcrest'] == ij]['cost'])

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
        options_height[i].drop(['Piping','StabilityInner','Section'],axis=1)
        options_geotechnical[i].drop(['Overflow','Section'],axis=1)
    return options_height,options_geotechnical

def SolveMIP(MIPModel):

    MixedIntegerSolution = MIPModel.solve()
    return MixedIntegerSolution