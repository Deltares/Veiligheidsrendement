import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Local\OntwerpenVeiligheidsrendement')
    traject = '16-3' + '.xlsx'
    case = 'partieel'
    times = [0,50]
    omegas = {}
    for i in times:
        plt.figure(figsize=(10,5),constrained_layout=True)
        omegas[i] = pd.read_excel(path.joinpath(traject),sheet_name = 'Voor T=' + str(i) + ' ' + case,header=1,usecols='A,P:S',index_col=0)
        p_1 = plt.bar(range(0, len(omegas[i])-1), omegas[i]['omega_overslag'][:-1],label='Overslag',color='r')
        p_1average = plt.axhline(omegas[i]['omega_overslag'][-1],xmin=0,xmax=len(omegas[i])-1,linestyle=':',color='r',label='totaal overslag')

        p_2 = plt.bar(range(0, len(omegas[i])-1), omegas[i]['omega_stabiliteit'][:-1], bottom=omegas[i]['omega_overslag'][:-1],label='Stabiliteit',color='g')
        p_2average = plt.axhline(omegas[i]['omega_stabiliteit'][-1], xmin=0, xmax=len(omegas[i]) - 1, linestyle=':',color='g',label='totaal stabiliteit')

        p_3 = plt.bar(range(0, len(omegas[i])-1), omegas[i]['omega_piping'][:-1], bottom=omegas[i]['omega_stabiliteit'][:-1],label='Piping',color='b')
        p_3average = plt.axhline(omegas[i]['omega_piping'][-1], xmin=0, xmax=len(omegas[i]) - 1, linestyle=':',color='b',label='totaal piping')

        if case == 'partieel':
            p_ingreep = plt.plot(np.argwhere(omegas[i]['ingreep?']==1),np.ones((len(np.argwhere(omegas[i]['ingreep?']==1)),1)),marker='o',linestyle='',color='k')
        plt.xlabel('Dijkvak')
        plt.xticks(range(0, len(omegas[i])-1), omegas[i].index[:-1], rotation='vertical')
        plt.ylabel(r'$\omega$ [-]')
        # plt.yscale('log')
        # plt.ylim((0.001,10))
        plt.title('Faalkansruimtefactoren voor T=' + str(i))
        plt.legend(loc='upper right')
        plt.savefig(path.joinpath('Faalkansruimtefactoren T=' + str(i) + ' ' + case + '.png'),dpi=300)
        # plt.savefig(path.joinpath('Faalkansruimtefactoren T=' + str(i) + ' ' + case + ' begrensd.png'),dpi=300)
        plt.close()

if __name__ == '__main__':
    main()
