## This script runs a multitude of HBN calculations for various locations read from an Excel sheet.
## It uses reference HBN calculation files which it copies many times.

import pandas as pd
import shutil
import fileinput
import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import subprocess
sys.path.append('../../src')
from probabilistic_tools.hydra_ring_scripts import runHydraRing

def getPrfl(fileName):
    # Get profile information from *.prfl Versie 4.0

    prfl = {}
    count_for = ''
    for line in fileinput.input(fileName):
        if 'VERSIE' in line:
            if line.split()[1] != '4.0':
                raise Exception('prfl moet versie 4.0 zijn')
        elif 'ID' in line:
            prfl['ID'] = line.split()[1]
        elif 'RICHTING' in line:
            prfl['RICHTING'] = line.split()[1]
        elif 'VOORLAND' in line:
            count_for = 'VOORLAND'
            count=0
            total_count = np.int32(line.split()[1])
            voorland_array = np.empty((total_count,3))
        elif 'KRUINHOOGTE' in line:
            prfl['KRUINHOOGTE'] = line.split()[1]
        elif 'DIJK' in line:
            count_for = 'DIJK'
            count=0
            total_count = np.int32(line.split()[1])
            dijk_array = np.empty((total_count,3))
            pass
        elif 'DAM' in line: #damtype
            prfl['DAM'] = line.split()[1]
        elif 'DAMWAND' in line: #not used in Riskeer
            pass
        elif 'DAMHOOGTE' in line:
            prfl['DAMHOOGTE'] = line.split()[1]
        elif 'MEMO' in line:
            pass
        else:
            if count_for != '':
                #add points for voorland or dijk
                if count_for == 'VOORLAND':
                    voorland_array[count,:] = np.array(line.split(), dtype=np.float32)
                elif count_for == 'DIJK':
                    dijk_array[count,:] = np.array(line.split(), dtype=np.float32)
                count += 1
                if total_count == count:
                    count = 0
                    count_for = ''
            else:
                pass

    prfl['DIJK'] = dijk_array
    prfl['VOORLAND'] = voorland_array
    return prfl


def add_overtopping_discharges(input, critical_discharges):
    critical_discharges = critical_discharges.set_index('Situatie')
    mu = []
    sigma = []
    for count, i in input.iterrows():
        #select proper row:
        subset = critical_discharges.loc[i.gekb_situatie]
        if isinstance(subset, pd.Series):
            # 1 value found, so take mu and sigma
            mu.append(subset['mu'])
            sigma.append(subset['sigma'])
        else:
            #filter golfhoogteklasse:
            subset2 = subset.loc[subset.Golfhoogteklasse == i.Golfhoogteklasse]
            if subset2.shape[0]==0:
                if i.Golfhoogteklasse == 'Klasse < 1m':
                    subset2 = subset.loc[subset.Golfhoogteklasse == '0-1 meter'].squeeze()
                else:
                    raise Exception('Onbekende golfhoogteklasse {}')
            else:
                raise Exception('To be developed...')
            mu.append(subset2['mu'])
            sigma.append(subset2['sigma'])

    input['mu_qc'] = mu
    input['sigma_qc'] = sigma
    return input
def modifyOverflowfile(i, filespath, filename, refsql, refini,database_loc,NumericsSettings,lowerbound= -1,upperbound=2):
    newsql = filespath.joinpath(filename,filename + '.sql')
    newini = filespath.joinpath(filename,filename + '.ini')
    workdir = filespath.joinpath(filename)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    shutil.copy(refsql, newsql)
    shutil.copy(refini, newini)

    #input prfl data.
    prfl = getPrfl(filespath.parent.joinpath('Input_Profielen',i.prfl_bestandnaam))
    if prfl['DAM'] != '0': raise Exception('Profielbestand {} bevat een dam, dit is niet ondersteund'.format(i.prfl_bestandsnaam))

    # changes values in sql for Location, orientation, claculation method and variables
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('LocationName', i['Vaknaam']))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('HLCDID', str(i.LocationID)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        # sys.stdout.write(line.replace('ORIENTATION', str(i.prfl_oriëntatie)))
        sys.stdout.write(line.replace('ORIENTATION', str(prfl['RICHTING'])))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('TimeIntegration', str(i.TimeIntegrationSchemeID)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hmin', str(i.prfl_dijkhoogte+lowerbound)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hmax', str(i.prfl_dijkhoogte+upperbound)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('Hstep', str(0.25)))
    # insert the correct parameters
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('KRUINHOOGTE', str(i.prfl_dijkhoogte)))

    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('MU_QC', str(i.mu_qc/1000)))
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        sys.stdout.write(line.replace('SIGMA_QC', str(i.sigma_qc/1000)))


    #write profiles to file
    for j, line in enumerate(fileinput.input(newsql, inplace=1)):
        if 'INSERTPROFILES' in line:
            #loop over points:
            for k in range(0,prfl['DIJK'].shape[0]):
                sys.stdout.write('INSERT INTO [Profiles] VALUES ({:d}, {:d}, {:.3f}, {:.3f});'.format(1, k+1, prfl['DIJK'][k,0], prfl['DIJK'][k,1]) + '\n')
        elif 'INSERTCALCULATIONPROFILE' in line:
            #loop over points:
            for k in range(0,prfl['DIJK'].shape[0]):
                sys.stdout.write('INSERT INTO [CalculationProfiles] VALUES ({:d}, {:d}, {:.3f}, {:.3f}, {:.3f});'.format(1, k+1, prfl['DIJK'][k,0], prfl['DIJK'][k,1], prfl['DIJK'][k,2]) + '\n')
        elif 'INSERTFORELANDGEOMETRY' in line:
            #loop over points:
            for k in range(0,prfl['VOORLAND'].shape[0]):
                sys.stdout.write('INSERT INTO [FORELANDS] VALUES ({:d}, {:d}, {:.3f}, {:.3f});'.format(1, k+1, prfl['VOORLAND'][k,0], prfl['VOORLAND'][k,1]) + '\n')
        elif 'INSERTBREAKWATER' in line:
            pass
        else:
            sys.stdout.write(line)

    #TODO profiel goed in bestand zetten.
    # change values in ini
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('DIJKPAAL', i.Vaknaam))
    for j, line in enumerate(fileinput.input(newini, inplace=1)):
        sys.stdout.write(line.replace('DATABASEPATH', str(database_loc)))
    return newini

#First we generate all the input files
def main():
    PATH = Path(r'c:\PilotWSRL\Overslag')

    #Lees de locaties die moeten worden bekeken. Dit is standaarduitvoer uit de routine die de invoerbestanden maakt.
    inputfile = PATH.joinpath('GEKBdata.csv')

    #refereer naar basisbestanden met specifieke keys
    refsql = PATH.joinpath('Overslag_basis','sql_reference.sql')
    refini = PATH.joinpath('Overslag_basis','ini_reference.ini')

    #refereer naar de database die moet worden gebruikt.
    # database_locs = ['DatabaseOntwerp','DatabaseWBI']
    database_locs = ['DatabaseWBI','DatabaseOntwerp']

    for dbloc in database_locs:
        input = pd.read_csv(inputfile)
        database_loc = PATH.joinpath(dbloc)

        #lees data voor config voor Numerics en TimeIntegration
        configfile = list(database_loc.glob("*.config.sqlite"))[0]
        cnx = sqlite3.connect(configfile)
        TimeIntegrationTable = pd.read_sql_query("SELECT * FROM TimeIntegrationSettings", cnx)
        TimeIntegrationTable = TimeIntegrationTable.loc[TimeIntegrationTable.CalculationTypeID==6][['LocationID','TimeIntegrationSchemeID']]

        NumericsTable = pd.read_sql_query("SELECT * FROM NumericsSettings", cnx)
        NumericsTable = NumericsTable.loc[NumericsTable.MechanismID==101]

        input = input.merge(TimeIntegrationTable,on='LocationID')
        #add critical overtopping discharge
        critical_discharges = pd.read_csv(refini.parent.joinpath('kritischeoverslagdebieten.csv'))
        input = add_overtopping_discharges(input,critical_discharges)
        results_dir = PATH.joinpath('Resultaten_{}'.format(database_loc.stem))
        for count, location in input.iterrows():
            ini_file = modifyOverflowfile(location, results_dir,location.Vaknaam,refsql,refini, database_loc,NumericsTable.loc[NumericsTable.LocationId == location.LocationID])
            runHydraRing(ini_file)
        # exit()

if __name__ == '__main__':
    main()
