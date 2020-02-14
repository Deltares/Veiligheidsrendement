from pathlib import Path
from Verification import Verification_SampledCases
from HelperFunctions import pareto_frontier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
        # case = 'test_test'
        StartSet = 16
        filepath = PATH.joinpath(case)
        if not filepath.joinpath('ParetoResults' + str(StartSet) + '.csv').exists():
            # Verification_SampledCases.BatchRunOptimization(filepath,run_MIP=True, run_Greedy = True, pareto_on=True,pareto_sets= 50, pareto_samples=500000)
            Verification_SampledCases.BatchRunOptimization(filepath,run_MIP=True, run_Greedy = False, pareto_on=False,pareto_sets= 50, pareto_samples=500000)

        # ParetoFront for selected case
        #load the results:
        GreedyResults = pd.read_csv(filepath.joinpath('TCs_Greedy.csv'))
        LCC_MIP = np.sum(pd.read_csv(filepath.joinpath('TakenMeasures_MIP.csv'))['LCC'])
        f1 = open(filepath.joinpath('OptimalTC_MIP.txt'),'r')
        TR_MIP = np.float(f1.read())-LCC_MIP
        f1.close()
        # ParetoResults = pd.read_csv(filepath.joinpath('ParetoResults.csv'))
        if MakeInset:
            extent = (1, 10, 1, 20)
        #compute ParetoFrontier
        [LCC_ParetoFront, TR_ParetoFront, index] = pareto_frontier(PATH=filepath, maxX=False, maxY=False)
        Factor = 1.e6
        if MakeInset: inset_data = np.array([])
        fig1,ax1 = plt.subplots()
        for file in filepath.iterdir():
            if file.is_file() and 'ParetoResults' in file.stem:
                data = pd.read_csv(file)
                if file.stem == 'ParetoResults0':
                    ax1.scatter(data['LCC'].values/Factor,data['TR'].values/Factor,marker='o',c='gray',s=1,label='Random samples')
                elif file.stem == 'ParetoResultsGreedy':
                    ax1.scatter(data['LCC'].values/Factor,data['TR'].values/Factor,marker='o',c='green',s=1,label='Random samples greedy')
                else:
                    ax1.scatter(data['LCC'].values/Factor,data['TR'].values/Factor,marker='o',c='gray',s=1)
                if MakeInset and file.stem != 'ParetoResultsGreedy':
                    dataset = np.array([data['LCC'].values/Factor,data['TR'].values/Factor]).T
                    inset_indices = np.argwhere(((data['LCC'].values/Factor>extent[0]) & (data['LCC'].values/Factor <extent[1]) & (data['TR'].values/Factor>extent[2]) & (data['TR'].values/Factor<extent[3])))
                    dataset = dataset[inset_indices[:,0],:]
                    inset_data = np.concatenate((inset_data, dataset)) if inset_data.size else dataset
                else:
                    inset_data_greedy = np.array([data['LCC'].values/Factor,data['TR'].values/Factor]).T

        ax1.plot(np.divide(LCC_MIP,Factor),np.divide(TR_MIP,Factor),'b',marker='d',markersize=14, label='Integer optimal solution')
        # make radius line for optimum:
        total = TR_MIP + LCC_MIP
        x = np.arange(0, total, 10000)
        y = total - x
        ax1.plot(x/Factor, y/Factor, linestyle=':', color='blue', label='Equal optimal Total Cost')

        ax1.plot(np.divide(LCC_ParetoFront,Factor),np.divide(TR_ParetoFront,Factor),'k', label='Pareto Frontier')

        ax1.plot(GreedyResults['LCC'].values/Factor,GreedyResults['TR'].values/Factor,'r',marker = 'o', alpha=0.5, label='Greedy search path')
        ind = np.argmin(np.add(GreedyResults['LCC'].values,GreedyResults['TR'].values))
        ax1.plot(GreedyResults['LCC'].values[ind]/Factor,GreedyResults['TR'].values[ind]/Factor,'r',marker = 'o', markersize = 10, label='Greedy search '
                                                                                                                                                   'optimum')
        ax1.set_ylim(top=1.e8/Factor, bottom=0)
        ax1.set_xlim(left=0)
        if Factor == 1.e6:
            ax1.set_xlabel('LCC in M€')
            ax1.set_ylabel('Total Risk in M€')
        ax1.grid()
        plt.legend()

        # inset axes....

        axins = ax1.inset_axes([0.2, 0.2, 0.4, 0.4])
        # mark_inset(ax1,axins,loc1=2,loc2=4,fc="none",ec='0.5')
        #add data
        axins.scatter(inset_data[:,0],inset_data[:,1],marker='o',c='gray',s=1)
        axins.scatter(inset_data_greedy[:,0],inset_data_greedy[:,1],marker='o',c='green',s=1)
        axins.plot(np.divide(LCC_MIP,Factor),np.divide(TR_MIP,Factor),'b',marker='d',markersize=12)
        axins.plot(x, y, linestyle=':', color='blue', label='Equal optimal Total Cost')
        axins.plot(np.divide(LCC_ParetoFront, Factor), np.divide(TR_ParetoFront, Factor), 'k', label='Pareto Frontier')

        axins.plot(GreedyResults['LCC'].values / Factor, GreedyResults['TR'].values / Factor, 'r', marker='o', alpha=0.5, label='Greedy search path')
        ind = np.argmin(np.add(GreedyResults['LCC'], GreedyResults['TR']))
        axins.plot(GreedyResults['LCC'].values[ind] / Factor, GreedyResults['TR'].values[ind] / Factor, 'r', marker='o', markersize=10, label='Greedy search optimum')
        axins.plot(x/Factor, y/Factor, linestyle=':', color='blue', label='Equal optimal Total Cost')

        # sub region of the original image
        # x1, x2, y1, y2 = 1, 10, 1, 10
        axins.set_xlim(extent[0], extent[1])
        axins.set_ylim(extent[2], extent[3])
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.grid()
        ax1.indicate_inset_zoom(axins,edgecolor='b')
        plt.savefig(filepath.joinpath('ParetoResult.png'),bbox_inches='tight',dpi=600)

        plt.show()

    #



if __name__ == '__main__':
    main()