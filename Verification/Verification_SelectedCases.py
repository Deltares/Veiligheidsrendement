from pathlib import Path
from Verification import Verification_SampledCases
from HelperFunctions import pareto_frontier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#Here we run the special cases that we consider. Also a run for a Pareto case
#OverflowEqual
#OverflowDominant
#GeotechnicalEqual
def main():

    ParetoRun=True
    if ParetoRun:
        MakeInset = True
        #ParetoRun
        PATH = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\98_Papers\Journal\2020_SAFEGreedyMethod_CACAIE\Berekeningen\resultaten\ParetoFront')
        case = '06_016'
        # case = '08_001'
        # case = 'test_test'
        StartSet = 16
        filepath = PATH.joinpath(case)
        # if not filepath.joinpath('ParetoResults' + str(StartSet) + '.csv').exists():
            # Verification_SampledCases.BatchRunOptimization(filepath,run_MIP=True, run_Greedy = True, pareto_on=True,pareto_sets= 50, pareto_samples=500000)
        GreedyResults = pd.read_csv(filepath.joinpath('TCs_Greedy.csv'))
        # LCClist = list(GreedyResults['LCC'].values + 100)[1:]
        LCClist = list(np.linspace(1e5, 22e6, 15))
        # LCClist = [1e6, 3e6, 5e6]
        Verification_SampledCases.BatchRunOptimization(filepath,run_MIP=False, run_Greedy = False, pareto_on=False,paretoLCCs=LCClist)

        # ParetoFront for selected case
        #load the results:
        LCC_MIP = np.sum(pd.read_csv(filepath.joinpath('TakenMeasures_MIP.csv'))['LCC'])
        ParetoResults = pd.read_csv(filepath.joinpath('TCs_Pareto.csv'))

        f1 = open(filepath.joinpath('OptimalTC_MIP.txt'),'r')
        TR_MIP = np.float(f1.read())-LCC_MIP
        f1.close()
        # ParetoResults = pd.read_csv(filepath.joinpath('ParetoResults.csv'))
        if MakeInset:
            extent = (4, 8, 0, 10)
        Factor = 1.e6
        if MakeInset: inset_data = np.array([])
        color = sns.cubehelix_palette(n_colors=3, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3)
        color = ['b','k','r']
        fig1,ax1 = plt.subplots()


        # make radius line for optimum:
        total = TR_MIP + LCC_MIP
        x = np.arange(0, total, 10000)
        y = total - x


        ax1.plot(GreedyResults['LCC'].values/Factor,GreedyResults['TR'].values/Factor,color=color[2],marker = 'o', label='Greedy search path')
        ind = np.argmin(np.add(GreedyResults['LCC'].values,GreedyResults['TR'].values))
        ax1.plot(GreedyResults['LCC'].values[ind]/Factor,GreedyResults['TR'].values[ind]/Factor,color=color[2],marker = 'o', linestyle = '', markersize = 10, label='Greedy search '
                                                                                                                                                   'optimum')

        ax1.plot(np.divide(ParetoResults['LCC'],Factor),np.divide(ParetoResults['TR'],Factor),color = color[1], marker='P', markersize=6, linestyle='', label='Pareto Frontier')

        ax1.plot(np.divide(LCC_MIP,Factor),np.divide(TR_MIP,Factor),color=color[0],marker='d',markersize=8, label='Integer optimal solution')
        ax1.plot(x/Factor, y/Factor, linestyle=':',color=color[0], label='Equal optimal Total Cost')
        ax1.set_ylim(top=.6e8/Factor, bottom=0)
        ax1.set_xlim(left=0,right=9)
        if Factor == 1.e6:
            ax1.set_xlabel('LCC in M€')
            ax1.set_ylabel('Total Risk in M€')
        ax1.grid()
        plt.legend()

        # inset axes....
        if MakeInset:
            axins = ax1.inset_axes([0.33, 0.2, 0.4, 0.4])
            # mark_inset(ax1,axins,loc1=2,loc2=4,fc="none",ec='0.5')
            #add data
            # axins.scatter(inset_data[:,0],inset_data[:,1],marker='o',c='gray',s=1)
            # axins.scatter(inset_data_greedy[:,0],inset_data_greedy[:,1],marker='o',c='green',s=1)
            axins.plot(x, y, linestyle=':', color=color[0], label='Equal optimal Total Cost')

            axins.plot(GreedyResults['LCC'].values / Factor, GreedyResults['TR'].values / Factor,color= color[2], marker='o', markersize=8, label='Greedy search path')
            ind = np.argmin(np.add(GreedyResults['LCC'], GreedyResults['TR']))
            axins.plot(np.divide(ParetoResults['LCC'],Factor),np.divide(ParetoResults['TR'],Factor),color=color[1], marker='P', markersize=6, linestyle = '', label='Pareto Frontier')
            axins.plot(GreedyResults['LCC'].values[ind] / Factor, GreedyResults['TR'].values[ind] / Factor,color= color[2], marker='o', linestyle = '', markersize=12, label='Greedy search optimum')
            axins.plot(np.divide(LCC_MIP,Factor),np.divide(TR_MIP,Factor),color=color[0],marker='d',markersize=8)
            axins.plot(x/Factor, y/Factor, linestyle=':', color=color[0], label='Equal optimal Total Cost')

            # sub region of the original image
            # x1, x2, y1, y2 = 1, 10, 1, 10
            axins.set_xlim(extent[0], extent[1])
            axins.set_ylim(extent[2], extent[3])
            axins.set_xticklabels('')
            axins.set_yticklabels('')
            axins.grid(linewidth=0.5)
            ax1.indicate_inset_zoom(axins,edgecolor='k',linestyle='--')
        plt.savefig(filepath.joinpath('ParetoResult.png'),bbox_inches='tight',dpi=600)

        plt.show()

    #



if __name__ == '__main__':
    main()