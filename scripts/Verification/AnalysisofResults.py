from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from HelperFunctions import calcRsquared
from Verification.Verification_SampledCases import BatchRunOptimization
import seaborn as sns
def computeMetrics(data,PATH, rel_tol=1e-4,write_others=True):
    N_cases = data.shape[0]

    TC = {}
    TC['TC global optima found'] = np.sum(data['relative error TC'] < rel_tol)/ N_cases
    TC['TC error > 1 perc'] = np.sum(data['relative error TC'] > 0.01)/ N_cases
    TC['TC largest relative error'] = np.max(data['relative error TC'])
    TC['TC(p(dTC>5%))'] = np.percentile(data['relative error TC'],95)
    TC['TC average difference'] = np.average(np.divide(np.abs(data['TC Greedy']-data['TC MIP']),data['TC MIP']))
    TC['TC Rsquared'] = calcRsquared(data['TC Greedy'], data['TC MIP'])

    LCC = {}
    LCC['LCC global optima found'] = np.sum(data['relative error LCC'] < rel_tol)/ N_cases
    LCC['LCC error > 1 perc'] = np.sum(data['relative error LCC'] > 0.01)/ N_cases
    LCC['LCC largest relative error'] = np.max(data['relative error LCC'])
    LCC['LCC(p(dLCC>5%))'] = np.percentile(data['relative error LCC'],95)
    LCC['LCC average difference'] = np.average(np.divide(np.abs(data['LCC Greedy'] - data['LCC MIP']), data['LCC MIP']))
    LCC['LCC Rsquared'] = calcRsquared(data['LCC Greedy'], data['LCC MIP'])
    if write_others:
        case_no =[]
        run_no =[]
        missing_measures_differences = []
        type_differences = []
        year_differences = []

        berms_differences = []
        crests_differences =[]
        #compare measures
        for i in data.iterrows():
            MIP_measures = pd.read_csv(PATH.joinpath('{:02d}'.format(i[1]['case name']), '{:03d}'.format(i[1]['run number']),'TakenMeasures_MIP.csv'))
            MIP_measures = MIP_measures.drop([MIP_measures.columns[0],'LCC'],axis=1)
            MIP_measures = MIP_measures.drop(MIP_measures.loc[MIP_measures['name'] == 'Do Nothing'].index, axis=0)
            if isinstance(MIP_measures['yes/no'].values[0],np.int64):
                MIP_measures['yes/no'] = MIP_measures['yes/no'].astype(str)


            Greedy_measures = pd.read_csv(PATH.joinpath('{:02d}'.format(i[1]['case name']), '{:03d}'.format(i[1]['run number']),'TakenMeasures_Optimal_Greedy.csv'))
            Greedy_measures = Greedy_measures.drop([Greedy_measures.columns[0],'LCC'],axis=1)
            if isinstance(Greedy_measures['yes/no'].values[0],np.int64):
                Greedy_measures['yes/no'] = Greedy_measures['yes/no'].astype(str)

            if not np.all(Greedy_measures.values == MIP_measures.values):
                case_no.append(i[1]['case name'])
                run_no.append(i[1]['run number'])
                if np.all(Greedy_measures['Section'].values == MIP_measures['Section'].values):
                    missing_measures_differences.append(0)
                else:
                    sections_not_in_MIP = list(set(Greedy_measures['Section']).difference(list(MIP_measures['Section'])))
                    sections_not_in_Greedy = list(set(MIP_measures['Section']).difference(list(Greedy_measures['Section'])))
                    missing_measures_differences.append(len(sections_not_in_Greedy)+len(sections_not_in_MIP))
                    if len(sections_not_in_Greedy) > 0:
                        for i in sections_not_in_Greedy:
                            MIP_measures = MIP_measures.drop(MIP_measures.loc[MIP_measures['Section'] == i].index,axis=0)
                    if len(sections_not_in_MIP) > 0:
                        for i in sections_not_in_MIP:
                            Greedy_measures = Greedy_measures.drop(Greedy_measures.loc[Greedy_measures['Section'] == i].index,axis=0)
                    # TODO take longest array, check if each element exists.
                    # Then sum the number of missing elements.
                    # write that value.
                    # delete elements that are not in both.
                    pass

                    # register the number of differences & throw out entries that are only in 1 dataframe
                if np.all(Greedy_measures['name'].values == MIP_measures['name'].values):
                    type_differences.append(0)
                    year_differences.append(0)

                else:
                    differencesyear = 0
                    differencestype = 0
                    for i in range(0, len(Greedy_measures['name'].values)):
                        # check the year
                        if Greedy_measures['name'].values[i][-4:] != MIP_measures['name'].values[i][-4:]:
                            differencesyear +=1
                        if Greedy_measures['name'].values[i][:-4] != MIP_measures['name'].values[i][:-4]:
                            differencestype +=1

                    year_differences.append(differencesyear)
                    type_differences.append(differencestype)

                if np.any(Greedy_measures['dberm'].values != MIP_measures['dberm'].values):
                    berms_differences.append(np.sum(Greedy_measures['dberm'].values != MIP_measures['dberm'].values))
                else:
                    berms_differences.append(0)

                if np.any(Greedy_measures['dcrest'].values != MIP_measures['dcrest'].values):
                    crests_differences.append(np.sum(Greedy_measures['dcrest'].values != MIP_measures['dcrest'].values))
                else:
                    crests_differences.append(0)
        results = pd.DataFrame(np.array([case_no, run_no, type_differences, year_differences, berms_differences, crests_differences]).T,
                     columns=['case', 'run', 'type differences', 'year differences', 'berms differences', 'crests differences'])
    else:
        results = []
    return TC, LCC, results

def writeOutputMetrics(data,PATH,rel_tol=1e-5,setting = 'per_case'):
    '''setting can be per case, per measureset, per system size'''
    caseset = pd.read_csv(PATH.joinpath('CaseSet.csv'))
    TC_all = []
    LCC_all = []
    others_all = []
    if setting == 'per_case':
        column = 'case name'
    elif setting == 'per_set':
        column = 'measure set'
    elif setting == 'per_size':
        caseset_reduced = caseset[['CaseNumber', 'Sections']]
        caseset_reduced.columns = ['case name', 'number of sections']
        data = pd.merge(data, caseset_reduced, on='case name')
        column = 'number of sections'

    # first add metrics for all casenumbers
    if not setting == 'per_overall':
        for i in np.unique(data[column]):
            relevant_data = data.loc[data[column] == i]
            TC, LCC, others = computeMetrics(relevant_data, PATH, rel_tol=rel_tol)
            TC_all.append(TC)
            LCC_all.append(LCC)
            others_all.append(others)

    else:
        write_others=False
        TC, LCC, others = computeMetrics(data, PATH, rel_tol=rel_tol,write_others=write_others)
        others_all.append(others)
        TC_all.append(TC)
        LCC_all.append(LCC)

    TC_all = pd.DataFrame(TC_all)
    LCC_all = pd.DataFrame(LCC_all)
    if not setting == 'per_overall':
        sets = np.unique(data[column])
        others_results = pd.DataFrame(columns=others_all[0].columns)
        different_cases = []
        for i in range(0, len(others_all)):
            line = np.sum(others_all[i], axis=0)
            others_results = others_results.append(line, ignore_index=True)
            different_cases.append(others_all[i].shape[0])
        others_results = others_results.drop(['case', 'run'], axis=1)
        others_results.insert(0,'case differences',different_cases)
        others_results.insert(0,setting.split('_')[1],sets)
        #optional for debugging:
        # for i in others_all:
        #     others_per_case = pd.concat((others_per_case, i))

        output_cases = pd.concat((TC_all, LCC_all, others_results), axis=1).reindex(TC_all.index)
        my_column = output_cases.pop(setting.split('_')[1])
        output_cases.insert(0, my_column.name, my_column)
    else:
        output_cases = pd.concat((TC_all, LCC_all), axis=1).reindex(TC_all.index)

    output_cases.to_csv(PATH.joinpath('OutputMetrics_' + setting + '.csv'))
    return output_cases
def readSampledResults(PATH,caselist,print_large_diffs = False,rel_crit = 0.01,runtime_analysis=False):
    #LCC of both methods
    TC_MIP = []
    TC_Greedy = []
    LCC_MIP = []
    LCC_Greedy = []
    relativeTC =[]
    relativeLCC =[]
    large_diffs = []
    casenames = []
    setnames = []
    measuresets = []
    caseinfo = pd.read_csv(PATH.joinpath('CaseSet.csv'),delimiter=',')
    runtimeGreedy = []
    runtimeMIP1 = []
    runtimeMIP2 = []
    for i in caselist:
        measuresets.append(caseinfo.loc[caseinfo['CaseNumber']==np.int32(i[0])]['MeasureSet'].values[0])
        #read objective
        f1 = open(PATH.joinpath(i[0],i[1],'OptimalTC_MIP.txt'),'r')
        TC_MIP.append(np.float(f1.read()))
        f1.close()
        f2 = open(PATH.joinpath(i[0],i[1],'OptimalTC_Greedy.txt'),'r')
        TC_Greedy.append(np.float(f2.read()))
        f2.close()
        if runtime_analysis:
            f3 = open(PATH.joinpath(i[0],i[1],'Greedy_computation_time.txt'))
            runtimeGreedy.append(np.float(f3.read()))
            f3.close()
            f4 = open(PATH.joinpath(i[0],i[1],'MIP_computation_time.txt'))
            MIPstr = f4.read()
            runtimeMIP1.append(np.float(MIPstr[0:MIPstr.find('.')+4]))
            runtimeMIP2.append(np.float(MIPstr[MIPstr.find('.')+4:]))
        relativeTC.append(np.abs((TC_Greedy[-1]-TC_MIP[-1])/TC_MIP[-1]))
        LCC_MIP.append(np.sum(pd.read_csv(PATH.joinpath(i[0],i[1],'TakenMeasures_MIP.csv'))['LCC']))
        LCC_Greedy.append(np.sum(pd.read_csv(PATH.joinpath(i[0],i[1],'TakenMeasures_Optimal_Greedy.csv'))['LCC']))
        relativeLCC.append(np.abs((LCC_Greedy[-1]-LCC_MIP[-1])/LCC_MIP[-1]))

        if print_large_diffs:
            if relativeTC[-1] > rel_crit:
                print(i[0] + 'and' + i[1] + ' has a difference of ' + str(relativeTC[-1]))
                large_diffs.append(i)


        casenames.append(i[0])
        setnames.append(i[1])
    if not print_large_diffs:
        large_diffs = None
    if runtime_analysis:
        data = pd.DataFrame(np.array([casenames, setnames, measuresets, TC_Greedy, TC_MIP, relativeTC, LCC_Greedy, LCC_MIP, relativeLCC,runtimeGreedy,runtimeMIP1,
                                      runtimeMIP2]).T,
                            columns=['case name', 'run number', 'measure set', 'TC Greedy', 'TC MIP', 'relative error TC', 'LCC Greedy', 'LCC MIP', 'relative error '
                                                                                                                                                    'LCC',
                                     'runtime Greedy', 'runtime MIP initialization', 'runtime MIP solve'])
    else:
        data = pd.DataFrame(np.array([casenames,setnames,measuresets,TC_Greedy,TC_MIP,relativeTC,LCC_Greedy,LCC_MIP,relativeLCC]).T,
                        columns=['case name','run number','measure set','TC Greedy','TC MIP','relative error TC','LCC Greedy','LCC MIP', 'relative error LCC'])
    data.to_csv(PATH.joinpath('Results.csv'))
    return data, large_diffs

def plot_data_points(PATH, data,type = 'TC',rsquared=True,subset = False,to_axis = False,ax_handle=None,markercolor='r'):
    x = data[type + ' Greedy'].values.astype(np.float32)
    y = data[type + ' MIP'].values.astype(np.float32)
    if not to_axis:
        fig1, ax1 = plt.subplots()
        ax1.plot(np.divide(x,1e6), np.divide(y,1e6), 'or', markersize = 1, label='Samples')
        coords = [np.min(np.floor(np.divide(np.array([x, y]), 1e6))) * 1e6, np.max(np.ceil(np.divide(np.array([x, y]), 1e6))) * 1e6]
        coords = [np.min(np.floor(np.divide(np.array([x, y]), 1e6))), np.max(np.ceil(np.divide(np.array([x, y]), 1e6)))]
        ax1.set_xlim(left=coords[0], right=coords[1])
        ax1.set_ylim(bottom=coords[0], top=coords[1])
        ax1.plot(coords, coords, 'k--')
        ax1.set_xlabel(type + ' Greedy in M€')
        ax1.set_ylabel(type + ' MIP in M€')
        ax1.set_title('Comparison of ' + type + ' for Greedy search and MIP')
        ax1.legend()
        if rsquared:
            rsq = calcRsquared(x, y)
            ax1.text(.85, .9, r'$R^2$ = ' + '{:.5f}'.format(rsq), horizontalalignment='right', verticalalignment='bottom', bbox=dict(fill=False), transform=ax1.transAxes)
        if subset:
            plt.savefig(PATH.joinpath('DataPlot_' + type + '_Subset_' + subset + '.png'),dpi=300,bbox_inches='tight')
        else:
            plt.savefig(PATH.joinpath('DataPlot_' + type + '.png'), dpi=300, bbox_inches='tight')
    else:
        ax_handle.plot(np.divide(x,1e6), np.divide(y,1e6),linestyle='', marker='o', color = markercolor, markerfacecolor=markercolor, markersize = 1, label='Samples')
        coords = [np.min(np.floor(np.divide(np.array([x, y]), 1e6))), np.max(np.ceil(np.divide(np.array([x, y]), 1e6)))]
        ax_handle.set_xlim(left=coords[0], right=coords[1])
        ax_handle.set_ylim(bottom=coords[0], top=coords[1])
        ax_handle.plot(coords, coords, 'k--',linewidth=0.5)
        ax_handle.set_xlabel(type + ' Greedy in €')
        ax_handle.set_ylabel(type + ' MIP in €')
        ax_handle.set_title(type)
        ax_handle.grid()
        # ax_handle.legend(loc=4)
        if rsquared:
            rsq = calcRsquared(x, y)
            ax_handle.text(.85, .9, r'$R^2$ = ' + '{:.5f}'.format(rsq), horizontalalignment='right', verticalalignment='bottom', bbox=dict(fill=False),
                           transform=ax_handle.transAxes)
        # return ax_handle

def cost_and_error(data,PATH,logaxis=True,TC_color='b',LCC_color='r'):
    color = sns.cubehelix_palette(n_colors=4, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3)
    color = ['r', '', '', 'b']
    TC_color = color[3]
    LCC_color = color[0]
    fig1 = plt.figure(constrained_layout=True)
    gs = fig1.add_gridspec(2, 2)
    ax_rolling = fig1.add_subplot(gs[:, 0])
    ax_rsq_TC = fig1.add_subplot(gs[0, -1])
    ax_rsq_LCC = fig1.add_subplot(gs[-1, -1])

    #first fill fig_rolling with 2 lines of rolling window
    topy = 5

    ax_rolling.grid('on')
    ax_rolling.set_xlabel('cost in M€')
    ax_rolling.set_ylabel('relative difference in %')
    ax_rolling.set_title('rel. difference')
    if logaxis:
        ax_rolling.set_yscale('log')
        ax_rolling.set_ylim(bottom=1e-2, top=topy)
        ax_rolling.set_xlim(left=0, right=250)
    else:
        ax_rolling.set_ylim(bottom=0, top=.5)
        ax_rolling.set_xlim(left=0, right=250)

    # add average relative error
    window = 20e6
    grid = np.arange(0, np.max(data['TC MIP'].values), step=1e6)
    errorTC = []
    errorLCC = []
    mean = []
    for i in range(0, len(grid)):
        errorTC.append(np.mean(
            data['relative error TC'].loc[(data['TC MIP'] > grid[i - 1] - window) & (data['TC MIP'] < grid[i] + window)]))
        errorLCC.append(np.mean(
            data['relative error LCC'].loc[(data['LCC MIP'] > grid[i - 1] - window) & (data['LCC MIP'] < grid[i] + window)]))
        mean.append(grid[i])
    ax_rolling.plot(np.divide(mean, 1e6), np.multiply(errorTC, 100), label='TC rolling avg. diff. (window = ' + str(np.int32(2 * window / 1e6)) + ' M€)',\
                                                                            color=TC_color)
    ax_rolling.plot(np.divide(mean, 1e6), np.multiply(errorLCC, 100), label='LCC rolling avg. diff. (window = ' + str(np.int32(2 * window / 1e6)) + ' M€)',
                    color=LCC_color)
    ax_rolling.legend(loc=1,fontsize='x-small')
    plot_data_points(PATH,data,type='TC',to_axis=True,ax_handle=ax_rsq_TC,markercolor=TC_color,rsquared=False)
    plot_data_points(PATH,data,type='LCC',to_axis=True,ax_handle=ax_rsq_LCC,markercolor=LCC_color,rsquared=False)
    plt.savefig(PATH.joinpath('RelativeErrorTCandLCC.png'),dpi=300,bbox_inches='tight')
    plt.show()




def main():
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Normal_cautious_f=1.5')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Overflow_cautious_f=1.5')
    # PATH = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\98_Papers\Journal\2020_SAFEGreedyMethod_CACAIE\Berekeningen\resultaten\AllCases_cautious_f=1.5')

    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Overflow_cautious_f=3')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Normal_cautious_f=3')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\AllCases_cautious_f=3')

    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Overflow_robust')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Normal_robust')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\AllCases_robust')

    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Overflow_combined')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\Batch_Normal_combined')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\AllCases_cautious_combined')

    PATH = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\98_Papers\Journal\2020_SAFEGreedyMethod_CACAIE\Berekeningen\uitgepakt\TestComputationTime')

    caselist = []
    rerun = False
    plots = False
    table1 = False
    plot_cost_vs_error = False
    results_read = False
    analyze_computation_time = True
    for mainpath in PATH.iterdir():
        if mainpath.is_dir() and mainpath.name != 'BaseData':
            for subpath in PATH.joinpath(mainpath).iterdir():
                caselist.append((mainpath.name, subpath.name))
        else:
            if mainpath.name == 'Results.csv':
                results_read = True

    if not results_read:
        data_computations, large_diffs = readSampledResults(PATH,caselist,runtime_analysis=analyze_computation_time)
    else:
        data_computations = pd.read_csv(PATH.joinpath('Results.csv'))

    # data_computations.to_excel(PATH.joinpath('ComputationData.xlsx'))

    ## Rerun all cases with a deviation in TC or LCC >0.005
    # filter data_computations
    if rerun:
        thresh = 0.005
        selected_cases = data_computations.loc[(data_computations['relative error TC'].astype(np.float)>thresh) | (data_computations['relative error LCC'].astype(
            np.float)> thresh)]
        selected_cases = selected_cases.loc[selected_cases['case name'].values.astype(np.int32)>19]
        print(str(len(selected_cases)) + ' cases for rerun')
        for i in selected_cases.iterrows():
            print('rerunning ' + str(i[1]['run number']) + ' of case ' + str(i[1]['case name']))
            GreedySettings = {'setting':'robust','f':1.,'BCstop':0.1}
            BatchRunOptimization(PATH.joinpath(str(i[1]['case name']),'{:03d}'.format(i[1]['run number'])), GreedySettings=GreedySettings, plot_on=False,pareto_on=False, run_MIP=False)
        data_computations_rerun, large_diffs_rerun = readSampledResults(PATH, caselist)
        data_computations_rerun.to_excel(PATH.joinpath('ComputationData_rerun.xlsx'))

    if plots:
        ##Compare data
        plot_data_points(PATH,data_computations,type='TC')
        plot_data_points(PATH,data_computations,type='LCC')

        ##Compare data of cases
        for i in data_computations['case name'].unique():
            plot_data_points(PATH,data_computations.loc[data_computations['case name']==i],type='LCC',subset= 'Case ' + str(i))
            plot_data_points(PATH,data_computations.loc[data_computations['case name']==i],type='TC',subset= 'Case ' + str(i))

        ##Compare data per measureset
        MeasureSet = pd.read_csv(PATH.joinpath('MeasureSet.csv'))

        for i in data_computations['measure set'].unique():
            plot_data_points(PATH,data_computations.loc[data_computations['measure set']==i],type='LCC',subset= 'MeasureSet ' + i)
            plot_data_points(PATH,data_computations.loc[data_computations['measure set']==i],type='TC',subset= 'MeasureSet ' + i)

    if table1:
        output_all  = writeOutputMetrics(data_computations, PATH, rel_tol=1e-5, setting= 'per_overall')
        output_cases = writeOutputMetrics(data_computations, PATH, rel_tol=1e-5, setting='per_case')
        output_sets  = writeOutputMetrics(data_computations, PATH, rel_tol=1e-5, setting='per_set')
        output_sizes  = writeOutputMetrics(data_computations, PATH, rel_tol=1e-5, setting='per_size')

    if plot_cost_vs_error:
        cost_and_error(data_computations,PATH,logaxis=False)

    if analyze_computation_time:
        #plot runtime vs number of sections
        fig, ax = plt.subplots()
        counts = []
        for case in caselist:
            count = 0
            for i in PATH.joinpath(case[0], case[1]).glob('**/*.xlsx'):
                count += 1
            print(count)
            counts.append(count)
        sec_array = np.array(counts)
        # for i in sections:
        #     sec_array = np.concatenate((sec_array, np.ones((nruns, 1)) * i))
        avgGreedy = []
        avgMIP1 = []
        avgMIP2 = []
        for i in np.unique(counts):
            avgGreedy.append(np.mean(data_computations['runtime Greedy'].loc[np.argwhere(sec_array == i)[:,0]]))
            avgMIP1.append(np.mean(data_computations['runtime MIP initialization'].loc[np.argwhere(sec_array == i)[:,0]]))
            avgMIP2.append(np.mean(data_computations['runtime MIP solve'].loc[np.argwhere(sec_array == i)[:,0]]))
        coloropts = {'n_colors': 2, 'start': 1.5, 'rot': 2, 'gamma': 1.5, 'hue': 1.0, 'light': 0.8, 'dark': 0.3}
        colors = sns.cubehelix_palette(**coloropts)
        ax.plot(sec_array, data_computations['runtime Greedy'][0:len(sec_array)], color=colors[0], linestyle = '', marker = 'o')
        ax.plot(sec_array, data_computations['runtime MIP initialization'][0:len(sec_array)], color=colors[1], linestyle = '', marker = 'd')
        ax.plot(sec_array, data_computations['runtime MIP solve'][0:len(sec_array)], color=colors[1], linestyle = '', marker = 'o')
        ax.plot(np.unique(counts), avgGreedy, color=colors[0], label='Greedy solve')
        ax.plot(np.unique(counts), avgMIP1, color=colors[1], linestyle='--', label='MIP init')
        ax.plot(np.unique(counts), avgMIP2, color=colors[1], linestyle= ':', label='MIP solve')
        ax.plot(np.unique(counts), np.add(avgMIP2,avgMIP1), color=colors[1], label='MIP total')
        ax.set_xlabel('no of sections')
        ax.set_ylabel('time elapsed [in s]')
        ax.set_title('Runtime')
        ax.set_xlim(left=5, right=15)
        ax.legend()
        ax.grid()
        # ax.set_yscale('log')
        plt.savefig(PATH.joinpath('runtime_linear.png'),dpi=300,bbox_anchor='tight',format='png')
        ax.set_ylim(bottom=0, top=60)
        plt.savefig(PATH.joinpath('runtime2_linear.png'),dpi=300,bbox_anchor='tight',format='png')

if __name__ == '__main__':
    main()

# import numpy as np
# import scipy.stats
#
#
# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
#     return m, np.max([m - h, 0]), m + h
#
#
# x = data_computations['TC MIP'].values
# y = data_computations['relative error TC'].values
# interval = 5e6
# ranges = np.arange(0, np.ceil(np.max(x) / interval) * interval, interval)
# mid = np.empty((len(ranges) - 1,))
# low = np.empty((len(ranges) - 1,))
# upper = np.empty((len(ranges) - 1,))
# x_coord = np.empty((len(ranges) - 1,))
# for i in range(1, len(ranges)):
#     mid[i - 1], low[i - 1], upper[i - 1] = mean_confidence_interval(y[np.argwhere((x > ranges[i - 1]) & (x < ranges[i]))])
#     x_coord[i - 1] = np.mean([ranges[i - 1:i]])
# plt.plot(x_coord, mid, 'r')
# plt.plot(x_coord, low, 'r--')
# plt.plot(x_coord, upper, 'r--')
# plt.scatter(x, y, c='b')
# plt.yscale('log')



