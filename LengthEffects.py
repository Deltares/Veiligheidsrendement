## This script is to make a few figures for the length effect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Local\OntwerpenVeiligheidsrendement')
    filename = 'lengte-effect.xlsx'
    # sheet = 'piping 16-3'
    sheet = 'stab 16-3'
    data = pd.read_excel(path.joinpath(filename), sheet_name=sheet, header=2, usecols='A:I')
    case = 'zonder b'
    # case = 'met b'
    # mechanisme = 'piping'
    mechanisme = 'stabiliteit'


    plt.figure(figsize=(10,5))
    ind = np.array(range(0, len(data)))
    pf_init = plt.bar(ind, data['Faalkans'], label='Initiele faalkans', color='r',width=0.3)
    pf_eis = plt.axhline((1./10000),ind[0],ind[-1],linestyle = '--', color='k',label=r'$P_{eis}$')
    if (case == 'zonder b') & (mechanisme == 'piping'):
        pf_default_1 = plt.bar(ind+0.3, data['Standaard zonder b'], label='Versterkt a=0.9', color='g',width=0.3)
        pf_cum_default = plt.axhline(np.sum(data['Standaard zonder b']), ind[0], ind[-1], color='g', linestyle=':',linewidth=2)
        pf_adapted_1 = plt.bar(ind-0.3, data['Aangepast zonder b'], label='Versterkt a=0.4', color='b',width=0.3)
        pf_cum_adapted = plt.axhline(np.sum(data['Aangepast zonder b']), ind[0], ind[-1], color='b', linestyle=':',linewidth=2)
    elif (case == 'met b') & (mechanisme == 'piping'):
        pf_default_1 = plt.bar(ind + 0.3, data['Standaard met b'], label='Versterkt a=0.9', color='g', width=0.3)
        pf_cum_default = plt.axhline(np.sum(data['Standaard met b']), ind[0], ind[-1], color='g', linestyle=':', linewidth=2)
        pf_adapted_1 = plt.bar(ind - 0.3, data['Aangepast met b'], label='Versterkt a=0.25', color='b', width=0.3)
        pf_cum_adapted = plt.axhline(np.sum(data['Aangepast met b']), ind[0], ind[-1], color='b', linestyle=':', linewidth=2)
    elif (case == 'zonder b') & (mechanisme == 'stabiliteit'):
        pf_default_1 = plt.bar(ind+0.3, data['Standaard zonder b'], label='Versterkt a=0.033', color='g',width=0.3)
        pf_cum_default = plt.axhline(np.sum(data['Standaard zonder b']), ind[0], ind[-1], color='g', linestyle=':',linewidth=2)
        pf_adapted_1 = plt.bar(ind-0.3, data['Aangepast zonder b'], label='Versterkt a=0.7', color='b',width=0.3)
        pf_cum_adapted = plt.axhline(np.sum(data['Aangepast zonder b']), ind[0], ind[-1], color='b', linestyle=':',linewidth=2)
        pass
    elif (case == 'met b') & (mechanisme == 'stabiliteit'):
        pf_default_1 = plt.bar(ind+0.3, data['Standaard met b'], label='Versterkt a=0.033', color='g',width=0.3)
        pf_cum_default = plt.axhline(np.sum(data['Standaard met b']), ind[0], ind[-1], color='g', linestyle=':',linewidth=2)
        pf_adapted_1 = plt.bar(ind-0.3, data['Aangepast met b'], label='Versterkt a=0.06', color='b',width=0.3)
        pf_cum_adapted = plt.axhline(np.sum(data['Aangepast met b']), ind[0], ind[-1], color='b', linestyle=':',linewidth=2)
        pass
    pf_cum_init = plt.axhline(np.sum(data['Faalkans']),np.min(ind),np.max(ind),color='r',linestyle=':',label='cumulatief')
    plt.title('Resulterende faalkansen bij doorsnede-eisen met b = vaklengte')
    plt.xticks(ind,data['Section'],rotation='vertical')
    plt.legend()
    plt.ylim((1e-9,1e-1))
    # pf_adapted_2 = plt.bar(ind-0.3, data['Aangepast zonder b'], label='Versterkt default met b', color='b',width=0.3)

    # pf_adapted_2 = plt.bar(range(0, len(data)), data['Aangepast met b'], label='Versterkt aangepast met b', color='r')
    plt.yscale('log')
    plt.savefig(path.joinpath('Lengte-effecten' + sheet + ' ' + case + '.png'), dpi=300)

    print()



if __name__ == '__main__':
    main()