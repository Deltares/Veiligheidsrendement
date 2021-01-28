## This script runs a multitude of HBN calculations for various locations read from an Excel sheet.
## It uses reference HBN calculation files which it copies many times.

from HydraRing_scripts import runHydraRing, readDesignTable
import pandas as pd
import shutil
import fileinput
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
#First we generate all the input files
PATH = Path(r'c:\Users\wouterjanklerk\Documents\00_PhDGeneral\03_Cases\01_Rivierenland SAFE\Local\HydraRing\HBNberekeningen')
inputfile = PATH.joinpath('InputLocations.xlsx')
filespath = PATH
refsql = PATH.joinpath('SQLreference.sql')
refini = PATH.joinpath('inireference.ini')
input = pd.read_excel(inputfile)

#in sql change 'LocationName', HLCDID, ORIENTATION, TimeIntegration, Hmin, Hmax, Hstep
#copy reference sql and rename
writefiles=1
def writeFiles(i, filespath, filename, refsql, refini):
    newsql = filespath.joinpath(filename,filename + '.sql')
    newini = filespath.joinpath(filename,filename + '.ini')
    workdir = filespath.joinpath(filename)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    shutil.copy(refsql, newsql)
    shutil.copy(refini, newini)

    # changes values in sql
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('LocationName', i[1]['section name'] + '_' + i[1]['dijkpaal']))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('HLCDID', str(i[1]['HLCD ID'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('ORIENTATION', str(i[1]['orientation'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('TimeIntegration', str(i[1]['TimeIntegration'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hmin', str(i[1]['h_min'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hmax', str(i[1]['h_max'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hstep', str(i[1]['h_step'])))
    # change values in ini
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('SECTION_DIJKPAAL', i[1]['section name'] + '_' + i[1]['dijkpaal']))
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('DATABASEPATH', i[1]['db']))

    return newini
location_list = []
for i in input.iterrows():
    filename = i[1]['section name'] + '_' + i[1]['dijkpaal']
    if i[1]['calculate?'] == 1 and writefiles == 1:
        newini = writeFiles(i, filespath, filename, refsql, refini)
    else:
        print('Skipped ' + i[1]['section name'] + '_' + i[1]['dijkpaal'])

    if i[1]['calculate?'] == 1:
        runHydraRing(newini)
        location_list.append(filename)

        print('I ran ' + filename)
    else:
        pass

# locations = location_list
locations = ['DV01_VY094']
# # locations = ['DV01_VY094', 'DV02_VY090', 'DV02_VY091', 'DV03_VY078', 'DV04_VY074', 'DV04_VY077', 'DV05_VY069', 'DV10_VY030', 'DV10_VY034', 'DV11_VY023', 'DV12_VY018', 'DV13_VY014']
if not os.path.exists(filespath.joinpath('figures')):
    os.makedirs(filespath.joinpath('figures'))
for i in locations:
    table = readDesignTable(filespath.joinpath(i,'DESIGNTABLE_' + i + '.txt'))
    table_WL = readDesignTable(filespath.joinpath(i,'DESIGNTABLE_' + i + '_waterlevel.txt'))
    dijkpaal = i[-5:]
    line = input.loc[input['dijkpaal']==dijkpaal]
    # HBNHKV1l = line['HBN_HKV (1 l/m/s)'].values
    # HBNHKV5l = line['HBN_HKV (5 l/m/s)'].values
    # HBNHKV10l = line['HBN_HKV (10 l/m/s)'].values

    Kruin_nu = line['hcrest'].values
    p = 0.24* 1./10000
    plt.plot(table['Value'], table['Failure probability'],label='Hydraulic Load q = 1 l/m/s')
    plt.plot(table_WL['Value'], table_WL['Failure probability'],label='Water level')
    # plt.plot(HBNHKV1l[0],p,'ro',label='HKV 1 l/m/s')
    # plt.plot(HBNHKV5l[0],p,'yo',label='HKV 5 l/m/s')
    # plt.plot(HBNHKV10l[0],p,'bo',label='HKV 10 l/m/s')

    plt.vlines(Kruin_nu[0],0,1./1000,colors = 'tab:gray',linestyles='solid', label='Current crest')
    # plt.hlines(p,min(min(table['Value']), HBNHKV10l[0]),max(table['Value']),colors = 'k',linestyles='dotted', label='Overslagnorm')
    # plt.hlines(1./10000, min(min(table['Value']), HBNHKV10l[0]), max(table['Value']), colors='k', linestyles='dashdot', label='Trajectnorm')
    # plt.xlabel('Hoogte [m NAP]')
    plt.xlabel('Level [m ref]')
    # plt.ylabel('Faalkans (-/jaar)')
    plt.ylabel('Failure probability [-/year]')
    plt.yscale('log')
    plt.legend()
    plt.title(i)
    # plt.xlim((min(min(table['Value']), HBNHKV10l[0]), max(Kruin_nu[0]+1,HBNHKV1l[0])))
    plt.xlim((7.5, 8.5))
    plt.ylim((10e-10,1./1000))
    plt.grid()
    # plt.show()
    plt.savefig(filespath.joinpath('figures',i + 'Paper.png'), bbox_inches='tight')
    plt.close()

