import seaborn as sns
import shelve
import matplotlib.pyplot as plt
import pandas as pd
import os
from StrategyEvaluation import calcTrajectProb
from FiguresGeoRisk import plotkversusBeta
from scipy.stats import norm
from scipy.interpolate import interp1d

import numpy as np
from HelperFunctions import getMeasureTable

## GENERAL SETTINGS
timing = 1
save_beta_measure_plots = 0
years0= [0, 1, 10, 20, 40, 50]
mechanisms = ['Overflow', 'StabilityInner', 'Piping']
section = "DV02"
pad = 'd:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\98_Papers\\Conference\\GeoRisk_2019\\Calculations\\Input\\' #monitoring_at_' + section + '\\'
resultname = 'FINALRESULT.out'
language = 'EN'
print()
basecase = {}
cases = []
#
# for i in os.listdir(pad):
#     if i[0:4] == 'Case':
#         filename = pad + i + '\\' + resultname
#         my_shelf = shelve.open(filename)
#
#         if i == 'Case_9':
#             #basecase
#             for key in my_shelf:
#                 basecase[key]=my_shelf[key]
#             basecase['casename'] = 'base case'
#             my_shelf.close()
#         else:
#             #other case
#             cases.append(dict())
#             for key in my_shelf:
#                 cases[-1][key]=my_shelf[key]
#             cases[-1]['casename'] = i
#             my_shelf.close()
# print()
#
# #Plots:
#
# #Conductivity-beta for each case for the monitored section
# k, beta, p = plotkversusBeta(cases + [basecase],section,path=pad)
#
# #Compute the posterior reliability weighted with the scenario probabilities
# pfs = norm.cdf(-beta)
# pf_scens = np.multiply(pfs,p)
# beta_post = -norm.ppf(np.sum(pf_scens))
#
# print(beta_post)
#
#
#
# #All investment patterns for the cases in 1 figure. Highlight the investments at the monitored section.
#
# MeasureTable = getMeasureTable(cases[0]['TestCaseSolutions'])
#
# cases = cases + [basecase]
# plt.figure(101,figsize=(20,10))
# n_colors = len(cases)
# clrs = sns.color_palette(n_colors=n_colors)  # a list of RGB tuples
#
# count = 0
# for i in cases[:-1]:
#     i['TestCaseStrategyTC'].plotBetaCosts(i['TestCase'], path=pad, typ='multi', fig_id=101,labels = i['casename'],symbolmode='on',MeasureTable=MeasureTable,linecolor=clrs[count])
#     count += 1
# cases[-1]['TestCaseStrategyTC'].plotBetaCosts(i['TestCase'], path=pad, typ='multi',last='yes', fig_id=101, labels = cases[-1]['casename'],symbolmode='on',MeasureTable=MeasureTable,linecolor='k',linestyle='--')
#
# plt.figure(102,figsize=(20,10))
# count = 0
# for i in cases[:-1]:
#     i['TestCaseStrategyTC'].plotBetaCosts(i['TestCase'], t = 50, path=pad, typ='multi', fig_id=102,labels = i['casename'],symbolmode='on',MeasureTable=MeasureTable,linecolor=clrs[count])
#     count += 1
# cases[-1]['TestCaseStrategyTC'].plotBetaCosts(i['TestCase'], t = 50, path=pad, typ='multi',last='yes', fig_id=102, labels = cases[-1]['casename'],symbolmode='on',MeasureTable=MeasureTable,linecolor='k',linestyle='--')
# #Plot difference in costs to achieve target reliability (cVoI)

#
sections = ['DV02','DV03','DV04']
data = {}
data['cases'] = {}
data['basecases'] = {}
for ii in sections:
    cases = []
    basecases = []
    pad = 'd:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\98_Papers\\Conference\\GeoRisk_2019\\Calculations\\Input\\monitoring_at_' + ii + '\\'
    for i in os.listdir(pad):
        if i[0:4] == 'Case':
            filename = pad + i + '\\' + resultname
            my_shelf = shelve.open(filename)

            if i[-5:] == 'base':
                basecases.append(dict())
                #basecase
                for key in my_shelf:
                    basecases[-1][key]=my_shelf[key]
                basecases['casename'] = i
                my_shelf.close()
            else:
                #other case
                cases.append(dict())
                for key in my_shelf:
                    cases[-1][key]=my_shelf[key]
                cases[-1]['casename'] = i
                my_shelf.close()

    data['cases'][ii] = cases
    data['basecases'][ii] = basecases

def plotcVOI(data,threshold=None,t=0,pad = None):
    plt.figure(figsize=(8,4))
    #loop over the keys:
    LCC = {}
    LCCbase = {}
    VOI = {}
    LCC = {}
    count = 0

    for ii in list(data['cases'].keys()):
        colors = ['r','b','g']
        VOI[ii] = {}
        LCC[ii] = []
        LCCbase[ii] =[]
        cases=data['cases'][ii]
        basecase = data['basecases'][ii]
        VOI[ii]['Pscen'] = []
        for i in cases:
            if i['casename'] == 'Case_base': continue
            if i['casename'][-4:] != 'base': VOI[ii]['Pscen'].append(i['TestCase'].GeneralInfo['P_scen'])
            beta_traj = []
            #calc traject beta for each step
            for j in i['TestCaseStrategyTC'].Probabilities:
                betat,pt = calcTrajectProb(j,horizon=50)
                beta_traj.append(betat[t])
            LCCs = np.cumsum(i['TestCaseStrategyTC'].TakenMeasures['LCC'].values)
            if threshold == None:
                beta_target = -norm.ppf(i['TestCase'].GeneralInfo['Pmax'])
            else:
                beta_target = threshold
            LCC_func = interp1d(beta_traj,LCCs)
            ## PERHAPS ADD A TOTAL RISK TERM HERE!
            if i['casename'][-4:] == 'base':
                LCCbase[ii].append(np.float(LCC_func(beta_target)))
            else:
                LCC[ii].append(np.float(LCC_func(beta_target)))
        costfactor = 1e6
        plabel = 0.025
        raiselabel = 0.1
        VOI[ii]['cVoI'] = np.subtract(LCCbase[ii],LCC[ii])
        VOI[ii]['VoI'] = np.multiply(np.diff(np.concatenate((VOI[ii]['Pscen'],np.array([1])))),VOI[ii]['cVoI'])
        VOI[ii]['E(VoI)'] = np.sum(VOI[ii]['VoI'])
        plt.plot(VOI[ii]['Pscen'], VOI[ii]['cVoI']/costfactor, label = 'Section ' + ii[-2:],color=colors[count],marker='.')
        plt.text(plabel, raiselabel+VOI[ii]['cVoI'][0]/costfactor,
                 'E(VoI) = ' + str(np.round(VOI[ii]['E(VoI)']/costfactor,decimals=2)) + ' M€')
        # VOI[ii]['Pscen'][np.argmax(VOI[ii]['cVoI'])], VOI[ii]['cVoI'][np.argmax(VOI[ii]['cVoI'])]/costfactor,
        cases = []
        # plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        count +=1
        # betas = i for i in i['TestCaseStrategyTC'].Probabilities
                #get LCCs, get betas

    plt.xlabel(r'$P(k_{repr})$')
    plt.ylabel('VoI in M€')
    plt.title(r'Conditional VoI for different values of $k_{repr}$ for $\beta_{target} = $' + str(
        np.round(beta_target, decimals=2)))
    plt.legend(loc=5)
    plt.xlim((0, 1))
    plt.tight_layout()
    if pad == None:
        plt.show()
    else:
        plt.savefig(pad + '\\cVoI_beta=' + str(np.round(beta_target,decimals=2)) + '.png', bbox_inches='tight', dpi=300)
        plt.close()


    #Find the point where the safety standard has been crossed (potentially including interpolation so that it is properly comparable)
    #Get the LCC value at that point
    #get the LCC value from the base case
    #compute the CVOI
    #compute the VOI bij weighing
    #plot the CVOI
    #write the VOI as text in the same color next to the line
patth = 'd:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\98_Papers\\Conference\\GeoRisk_2019\\Calculations\\Input\\'
thresholds = [None, 2.5, 3, 3.5, 3.6, 3.8, 3.9, 4]
for i in thresholds:
    plotcVOI(data,threshold=i,pad=patth)


def plotkversusBeta(cases,section,betayear = 0,path = None):
    k = []
    beta = []
    p = []
    casenames = []
    # get the data
    for j in cases:
        for i in range(0,len(j['TestCase'].Sections)):
            if j['TestCase'].Sections[i].name == section:
                if j['casename'] != 'base case':
                    k.append(j['TestCase'].Sections[i].Reliability.Mechanisms['Piping'].Reliability['0'].Input.input['k'])
                    beta.append(j['TestCase'].Sections[i].Reliability.SectionReliability[str(betayear)]['Piping'])
                    p.append(j['TestCase'].GeneralInfo['P_scen'])
                    casenames.append(j['casename'])
                else:
                    k_base = j['TestCase'].Sections[i].Reliability.Mechanisms['Piping'].Reliability['0'].Input.input['k']
                    beta_base = j['TestCase'].Sections[i].Reliability.SectionReliability[str(betayear)]['Piping']

    k = np.asarray(k)
    beta = np.asarray(beta)
    p = np.asarray(p)
    p = np.diff(np.concatenate((p,np.array([1]))))

    #make the plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('k [m/s]')
    ax1.set_ylabel(r'Reliability index $\beta$')

    from scipy.stats import norm
    pfs = norm.cdf(-beta)
    pss = np.multiply(p, pfs)
    beta_expected = -norm.ppf(np.sum(pss))

    for i in range(0,len(casenames)):
        ax1.text(k[i]+.000005,beta[i],casenames[i])
    ax1.tick_params(axis='y', labelcolor='k')


    #part with scenario probs
    bar_width = 0.3*(np.amax(k)-np.amin(k))/9
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$P_{scenario}$', color = 'b')
    ax2.bar(k,p,width = bar_width, facecolor='b',alpha = 0.5,label='Scenario probability')
    ax2.tick_params(axis='y', labelcolor='b')

    #part with beta
    ax1.grid()
    ax1.plot(k,beta,'ok',label = r'$\beta$ per scenario')
    ax1.plot(k_base,beta_base,'sk', label = r'$\beta$ original')
    ax1.axvline(x=k_base, linestyle=':', color='k')
    # ax1.axhline(y=beta_base,linestyle=':', color='k')
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    # labs = [l.get_label() for l in legends]
    # plt.legend(legends,labs,loc=0)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(path + section + '_k_vs_beta.png')
    print()
    return k, beta, p
