## This script runs a multitude of HBN calculations for various locations read from an Excel sheet.
## It uses reference HBN calculation files which it copies many times.

# from HydraRing_scripts import runHydraRing, readDesignTable
import pandas as pd
import shutil
import fileinput
import sys
import os
# import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import subprocess
sys.path.append('../../src')
from probabilistic_tools.HydraRing_scripts import runHydraRing

def getPrfl(fileName):
    # Get profile information from *.prfl (type Hydra-NL)
    f = open(fileName, 'r')
    lines = f.readlines()

    prfl = {}
    count = 0

    for line in lines:

        if line.find('DAM') == 0 and line.find('DAMWAND') == -1 and line.find('DAMHOOGTE') == -1:
            split = line.split()
            prfl['dam'] = int(split[1])

        elif line.find('DAMHOOGTE') == 0:
            split = line.split()
            prfl['damhoogte'] = float(split[1])

        elif line.find('RICHTING') == 0:
            split = line.split()
            prfl['richting'] = float(split[1])

        elif line.find('KRUINHOOGTE') == 0:
            split = line.split()
            prfl['kruinhoogte'] = float(split[1])

        elif line.find('VOORLAND') == 0:
            split = line.split()
            prfl['voorland'] = int(split[1])

        elif line.find('DAMWAND') == 0:
            split = line.split()
            prfl['damwand'] = int(split[1])
            count_start = count

        elif line.find('MEMO') == 0:
            count_end = count

        count = count + 1

    count = count_start
    split = lines[count_start + 1].split()

    if count_start + 1 == count_end:
        prfl['xyrho'] = np.array([])
    else:
        prfl['xyrho'] = np.array([[float(split[0]), float(split[1]), float(split[2])]])

    while count < count_end - 1:
        count = count + 1
        split = lines[count].split()
        prfl['xyrho'] = np.append(prfl['xyrho'], [[float(split[0]), float(split[1]), float(split[2])]], axis=0)

    if prfl['damwand'] == 1:  # change to 1-op-1
        prfl['xyrho'] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [6.0, 6.0, 1.0]])

    return prfl

def modifyTPfile(i, filespath, filename, refsql, refini,database_locclimate=False):
    newsql = filespath.joinpath(filename,filename + '.sql')
    newini = filespath.joinpath(filename,filename + '.ini')
    workdir = filespath.joinpath(filename)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    shutil.copy(refsql, newsql)
    shutil.copy(refini, newini)

    # changes values in sql
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('LocationName', i['HR koppel']))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('HLCDID', str(i.LocationId)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('ORIENTATION', str(180)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('TimeIntegration', str(1))) #PAS OP!!!!!!!!!!
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        if not climate:
            sys.stdout.write(line.replace('Hmin', str(i['h_1:10_rtd'])))
        else:
            sys.stdout.write(line.replace('Hmin', str(i['h_1:10_rtd']+0.75)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        if not climate:
            sys.stdout.write(line.replace('Hmax', str(i['h_1:600.000'])))
        else:
            sys.stdout.write(line.replace('Hmax', str(i['h_1:600.000']+.75)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hstep', str(0.25)))

    # change values in ini
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('SECTION_DIJKPAAL', i.Vaknaam))
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('DATABASEPATH', str(database_loc)))
    return newini

#First we generate all the input files
def main():
    PATH = Path(r'c:\PilotWSRL')
    #lees input uit Locaties_voorStephan
    inputfile = PATH.joinpath('Locaties38-1_voorStephan.csv')

    refsql = PATH.joinpath('Toetspeil_basis','sql_reference.sql')
    refini = PATH.joinpath('Toetspeil_basis','ini_reference.ini')

    database_loc = PATH.joinpath('DatabaseOntwerp')
    configfile = list(database_loc.glob("*.config.sqlite"))[0]
    cnx = sqlite3.connect(configfile)
    config_table = pd.read_sql_query("SELECT * FROM TimeIntegrationSettings", cnx)
    input = pd.read_csv(inputfile)
    input = input.merge(config_table.loc[config_table.CalculationTypeID==0][['LocationID','TimeIntegrationSchemeID']],on='LocationID')
    results_dir = PATH.joinpath('ResultatenTPOntwerp')
    for count, location in input.iterrows():
        ini_file = modifyTPfile(location, results_dir,location.Vaknaam,refsql,refini, database_loc,climate=True)
        runHydraRing(ini_file)
        # exit()

if __name__ == '__main__':
    main()
