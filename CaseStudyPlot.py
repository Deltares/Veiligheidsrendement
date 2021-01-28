from pathlib import Path
import shelve
from DikeTraject import PlotSettings, getSectionLengthInTraject
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from HelperFunctions import getMeasureTable


def plotLCC(Strategies,traject,PATH=False,fig_size=(12,2),flip=False,title_in=False,subfig=False,greedymode = 'Optimal',color = False):
    #now for 2 strategies: plots an LCC bar chart
    cumlength, xticks1, middles = getSectionLengthInTraject(traject.Probabilities['Length'].loc[traject.Probabilities.index.get_level_values(1) == 'Overflow'].values)
    if not color:
        color = sns.cubehelix_palette(n_colors=4, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3)
    fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=fig_size, sharey='row',
                                  gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.08, 'left': 0.03, 'right': 0.98})
    for i in cumlength:
        ax.axvline(x=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    widths = traject.Probabilities['Length'].loc[traject.Probabilities.index.get_level_values(1) == 'Overflow'].values/2
    if greedymode == 'Optimal':
        GreedySolution = Strategies[0].OptimalSolution['LCC'].values/1e6
    elif greedymode == 'SafetyStandard':
        GreedySolution = Strategies[0].SatisfiedStandardSolution['LCC'].values/1e6
        print()
    ax.bar(np.subtract(middles,0.45*widths),GreedySolution,widths*0.9,color=color[0],label='Optimized')
    ax.bar(np.add(middles,0.45*widths),Strategies[1].FinalSolution['LCC'].values/1e6,widths*0.9,color=color[1],label='Target rel.')

    #make x-axis nice
    ax.set_xlim(left=0, right=np.max(cumlength))
    labels_xticks = []
    for i in traject.Sections:
        labels_xticks.append('S' + i.name[-2:])
    ax.set_xticks(middles)
    ax.set_xticklabels(labels_xticks)
    ax.tick_params(axis='x', rotation=90)
    #make y-axis nice
    LCCmax = np.max([Strategies[0].OptimalSolution['LCC'].values, Strategies[1].FinalSolution['LCC'].values]) / 1e6
    if LCCmax < 10: ax.set_ylim(bottom=0,top=np.ceil(LCCmax/2)*2)
    if LCCmax >=10: ax.set_ylim(bottom=0,top=np.ceil(LCCmax/5)*5)
    ax.set_ylabel('Cost in M€')
    ax.get_xticklabels()
    ax.tick_params(axis='both', bottom=False)

    #add a legend
    ax1.axis('off')
    ax.text(0, 0.8,'Total LCC Optimized = ' + '{:.0f}'.format(np.sum(GreedySolution)) +
            ' M€ \n' + 'Total LCC Target rel. = ' + '{:.0f}'.format(np.sum(Strategies[1].FinalSolution['LCC'].values / 1e6)) + ' M€',
            horizontalalignment='left', transform=ax.transAxes)
    if flip: ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(1.0001, 0.85)) #reposition!
    ax.grid(axis='y', linewidth=0.5, color='gray', alpha=0.5)
    if title_in:
        ax.set_title(title_in)
    plt.savefig(PATH.joinpath('LCC.png'),dpi=300, bbox_inches='tight', format='png')

def main():
    #initialize the case that we consider. We start with a small one, eventually we will use a big one.
    ## GENERAL SETTINGS
    PATH = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\SAFE_16-4_oktober_test')
    case = 'Case_cautious_f=1.5_bundling'

    ##PLOT SETTINGS
    plot_year = 2025
    rel_year = plot_year-2025
    directory = PATH.joinpath(case)
    filename = directory.joinpath('AfterStep1.out')
    my_shelf = shelve.open(str(filename))
    for key in my_shelf:
        TestCase = my_shelf[key]
    my_shelf.close()

    filename = directory.joinpath('AfterStep2.out')
    my_shelf = shelve.open(str(filename))
    for key in my_shelf:
        AllSolutions = my_shelf[key]
    my_shelf.close()

    filename = directory.joinpath('FINALRESULT.out')
    my_shelf = shelve.open(str(filename))
    for key in my_shelf:
        AllStrategies = my_shelf[key]
    my_shelf.close()
    AllStrategies[1].makeSolution(directory.joinpath('results', 'FinalMeasures_OI.csv'), type='Final')
    AllStrategies[0].makeSolution(directory.joinpath('results', 'FinalMeasures_TC.csv'), type='SatisfiedStandard')
    #pane 1: reliability in 2050
        #left: system reliability
        #right: all sections
    figsize = (12, 2)
    #color settings
    optimized_colors = {'n_colors': 5, 'start' : 1.5, 'rot' : 0.3, 'gamma' : 1.5, 'hue' : 1.0, 'light' : 0.8, 'dark' : 0.3}
    targetrel_colors = {'n_colors': 5, 'start' : 0.5, 'rot' : 0.3, 'gamma' : 1.5, 'hue' : 1.0, 'light' : 0.8, 'dark' : 0.3}

    # TestCase.plotAssessment(PATH=directory, fig_size=figsize, flip='on',
    #                         years = [rel_year], labels_limited=True, system_rel=True,
    #                         custom_name='Assessment_' + str(plot_year) + '.png', show_xticks=True,
    #                         title_in='(a) \n' + r'$\bf{Predicted~reliability~in~' + str(plot_year) + '}$')
    #
    # #pane 2: reliability in 2075, with Greedy optimization
    # TestCase.plotAssessment(PATH=directory, fig_size=figsize, flip='on',
    #                         years = [rel_year], labels_limited=True,system_rel=True,
    #                         custom_name='GreedyStrategy_' + str(plot_year) + '.png', reinforcement_strategy=AllStrategies[0], greedymode = 'SafetyStandard',
    #                         show_xticks=True, title_in='(c)\n' + r'$\bf{Optimized~investment}$ - Reliability in ' + str(plot_year),colors=optimized_colors)
    #
    # #pane 3: reliability in 2075, with Target Reliability Approach
    # TestCase.plotAssessment(PATH=directory, fig_size=figsize, flip='on',
    #                         years = [rel_year], labels_limited=True,system_rel=True,
    #                         custom_name='TargetReliability_' + str(plot_year) + '.png', reinforcement_strategy=AllStrategies[1],
    #                         show_xticks=True, title_in='(e) \n' + r'$\bf{Target~reliability~based~investment}$ -  Reliability in ' + str(plot_year),colors=targetrel_colors)
    #
    #pane 4: measures per dike section for Greedy
    AllStrategies[0].plotMeasures(traject=TestCase,PATH=directory,fig_size=figsize,crestscale=25.,
                                  show_xticks=False, flip=True,  greedymode = 'SafetyStandard',
                                  title_in='(b) \n' + r'Measures for optimized investment with large increase in hydraulic load',colors=optimized_colors)
    # #pane 5: measures per dike section for Target
    #
    # AllStrategies[1].plotMeasures(traject=TestCase,PATH=directory,fig_size=figsize,crestscale=25.,
    #                               show_xticks=False, flip=True,
    #                               title_in='(d) \n' + r'$\bf{Target~reliability~based~investment}$ - Measures',colors=targetrel_colors)

    # #pane 6: Investment costs per dike section for both
    twoColors = [sns.cubehelix_palette(**optimized_colors)[1],sns.cubehelix_palette(**targetrel_colors)[1]]
    plotLCC(AllStrategies,TestCase,PATH=directory,fig_size=figsize,flip=True,greedymode='SafetyStandard',
            title_in='(f) \n' + r'$\bf{LCC~of~both~approaches}$',color = twoColors)


    # LCC-beta for t=50
    MeasureTable = getMeasureTable(AllSolutions, language = 'EN',abbrev=True)
    figsize = (5,5)
    plt.figure(102, figsize=figsize)
    AllStrategies[0].plotBetaCosts(TestCase, t=rel_year,
                                   fig_id=102, symbolmode=True, markersize = 10, final_step = AllStrategies[0].OptimalStep,color=twoColors[0], series_name='Optimized ' \
                                                                                                                                                    'investment',
                                   MeasureTable=MeasureTable, beta_or_prob='beta', outputcsv=PATH,final_measure_symbols =False)
    AllStrategies[1].plotBetaCosts(TestCase, t=rel_year,
                                   fig_id=102, symbolmode=True, markersize = 10, color=twoColors[1], series_name='Target reliability based investment',
                                   MeasureTable=MeasureTable,last=True, beta_or_prob='beta', outputcsv=PATH,final_measure_symbols =True)
    plt.savefig(directory.joinpath('Priority order Beta vs LCC_' + str(plot_year) + '.png'),dpi=300,bbox_inches='tight',format='png')

if __name__ == '__main__':
    main()