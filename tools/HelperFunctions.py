from typing import Union

try:
    import cPickle as pickle
except:
    import pickle
import shelve
import copy
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import openturns as ot
from pandas import DataFrame, Series
from pathlib import Path
import numpy as np
import os
from shutil import copyfile, rmtree
## This .py file contains a bunch of functions that are useful but do not fit under any of the other .py files.

# write to file with cPickle/pickle (as binary)
def ld_writeObject(filePath,object):
    f=open(filePath,'wb')
    newData = pickle.dumps(object, 1)
    f.write(newData)
    f.close()

#Used to read a pickle file which can objects
def ld_readObject(filePath):
    # This script can load the pickle file so you have a nice object (class or dictionary
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

#Helper function to flatten a nested dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def drawAlphaBarPlot(resultList,xlabels = None, Pvalues = None, method = 'MCS', suppress_ind = None, title = None):
    #draw stacked bars for the importance factors of a list of openturns FORM and results
    idx = np.nonzero(Pvalues)[0][0]
    if method == 'MCS':
        labels = resultList[idx].getImportanceFactors().getDescription()
    elif method == 'FORM':
        labels = resultList[idx].getImportanceFactors(ot.AnalyticalResult.CLASSICAL).getDescription()

    alphas = []
    for i in range(idx, len(resultList)):
        alpha = list(resultList[i].getImportanceFactors(ot.AnalyticalResult.CLASSICAL)) if method == 'FORM' else list(resultList[i].getImportanceFactors())
        if suppress_ind != None:
            for ix in suppress_ind:
                alpha[ix] = 0.0
        alpha = np.array(alpha)/np.sum(alpha)
        alphas.append(alpha)

    alfas = DataFrame(alphas, columns=labels)
    alfas.plot.bar(stacked=True, label=xlabels)
    # print('done')
    if Pvalues != None:
        plt.plot(range(0, len(xlabels)), Pvalues, 'b', label='Fragility Curve')

    xlabels = ['{:4.2f}'.format(xlabels[i]) for i in range(0, len(xlabels))]
    plt.xticks(range(0, len(xlabels)), xlabels)
    plt.legend()
    plt.title(title)
    plt.show()
    #TO BE DONE: COmpare a reference case to Matlab

def calc_r_exit(h_exit,k,d_cover,D,wl,Lbase,Lachter,Lfore = 0):
    # k = 0.0001736111
    # h_exit = 2.5
    # wl = 6.489
    # Lbase = 36
    # Lachter = 5.65
    lambda2 = np.sqrt(((k* 86400) * D) * (d_cover / 0.01))
    #slight modification: foreshore is counted as dijkzate
    phi2 = h_exit + (wl - h_exit)*((lambda2*np.tanh(2000/lambda2))/(lambda2*np.tanh(Lfore/lambda2)+Lbase+Lachter+lambda2*np.tanh(2000/lambda2)))
    r_exit = (phi2-h_exit)/(wl-h_exit)
    return r_exit

def adaptInput(grid_data,monitored_sections,BaseCase):
    CasesGeoRisk = []
    for i, row in grid_data.iterrows():
        CasesGeoRisk.append(copy.deepcopy(BaseCase))
        # adapt k-value
        for j in CasesGeoRisk[-1].Sections:
            if j.name in monitored_sections:
                for ij in list(j.Reliability.Mechanisms['Piping'].Reliability.keys()):
                    j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['k'] = row['k']
                    wl = \
                    np.array(j.Reliability.Load.distribution.computeQuantile(1 - CasesGeoRisk[-1].GeneralInfo['Pmax']))[
                        0]
                    new_r_exit = calc_r_exit(j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['h_exit'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['k'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['d_cover'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['D'],
                                             wl,
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['Lvoor'],
                                             j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['Lachter'])
                    j.Reliability.Mechanisms['Piping'].Reliability[ij].Input.input['r_exit'] = new_r_exit

            CasesGeoRisk[-1].GeneralInfo['P_scen'] = row['p']                # print(str(row['k']) + '   ' + str(new_r_exit))
    return CasesGeoRisk

def replaceNames(TestCaseStrategy, TestCaseSolutions):
    TestCaseStrategy.TakenMeasures =TestCaseStrategy.TakenMeasures.reset_index(drop=True)
    for i in range(1, len(TestCaseStrategy.TakenMeasures)):
        # names = TestCaseStrategy.TakenMeasures.iloc[i]['name']
        #
        # #change: based on ID and get Names from new table.
        # if isinstance(names, list):
        #     for j in range(0, len(names)):
        #         names[j] = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names[j]].parameters['Name']
        # else:
        #     names = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names].parameters['Name']
        id = TestCaseStrategy.TakenMeasures.iloc[i]['ID']
        if isinstance(id, list):
            id = '+'.join(id)

        section = TestCaseStrategy.TakenMeasures.iloc[i]['Section']
        name = TestCaseSolutions[section].measure_table.loc[TestCaseSolutions[section].measure_table['ID'] == id]['Name'].values
        TestCaseStrategy.TakenMeasures.at[i, 'name'] = name
    return TestCaseStrategy


def get_measure_table(AllSolutions,language ='NL',abbrev=False):
    OverallMeasureTable = pd.DataFrame([], columns=['ID', 'Name'])
    for i in AllSolutions:
        OverallMeasureTable = pd.concat([OverallMeasureTable, AllSolutions[i].measure_table])
    OverallMeasureTable: Union[DataFrame, None, Series] = OverallMeasureTable.drop_duplicates(subset='ID')
    if (np.max(OverallMeasureTable['Name'].str.find('Grondversterking').values) > -1) and (language == 'EN'):
        OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Grondversterking binnenwaarts', 'Soil based')
        if abbrev:
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Grondversterking met stabiliteitsscherm', 'Soil based + SS')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Verticaal Zanddicht Geotextiel', 'VSG')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Zelfkerende constructie', 'DW')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Stabiliteitsscherm', 'SS')
        else:
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Grondversterking met stabiliteitsscherm', 'Soil inward + Stability Screen')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Verticaal Zanddicht Geotextiel', 'Vertical Sandtight Geotextile')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Zelfkerende constructie', 'Diaphragm Wall')
            OverallMeasureTable['Name'] = OverallMeasureTable['Name'].str.replace('Stabiliteitsscherm', 'Stability Screen')
    return OverallMeasureTable

#this is a more generic function to read and write data from and to a shelve. But it is not implemented fully:
# TODO implement DataAtShelve instead of (un)commenting snippets of code
def DataAtShelve(dir, name, objects = None, mode = 'write'):
    if mode == 'write':
        #make shelf
        my_shelf = shelve.open(str(dir.joinpath(name)), 'n')
        for i in objects.keys():
            my_shelf[i] = objects[i]
        my_shelf.close()
    elif mode == 'read':
        # open shelf
        my_shelf = shelve.open(str(dir.joinpath(name)))
        keys = []
        for key in my_shelf:
            locals()[key]=my_shelf[key]
            keys.append(key)
        my_shelf.close()
        if len(keys) == 1:
            return locals()[keys[0]]

def id_to_name(ID, MeasureTable):
    return MeasureTable.loc[MeasureTable['ID']==ID]['Name'].values[0]

def flatten(l): #flatten a list
  out = []
  if len(l)>0:
      for item in l:
        if isinstance(item, (list, tuple)):
          out.extend(flatten(item))
        else:
          out.append(item)
  return out

def pareto_frontier(Xs=False, Ys=False,PATH=False, maxX = True, maxY = True):
    if PATH:
        Xs = np.array([])  # LCC
        Ys = np.array([])  # TR
        # read info from path
        for file in PATH.iterdir():
            if file.is_file() and 'ParetoResults' in file.stem:
                data = pd.read_csv(file)
                Xs = np.concatenate((Xs, data['LCC'].values))
                Ys = np.concatenate((Ys, data['TR'].values))
    elif (isinstance(Xs,bool) or isinstance(Ys,bool)):
        raise IOError('No input provided')
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    index_order = np.argsort(Xs)
    p_front = [myList[0]]
    index = [index_order[0]]
    count = 1
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
                index.append(index_order[count])
        else:
            if pair[1] < p_front[-1][1]:
                p_front.append(pair)
                index.append(index_order[count])
            elif pair[1] == p_front[-1][1]:
                if pair[0] < p_front[-1][0]:
                    p_front.append(pair)
                    index.append(index_order[count])
        count +=1
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY, index

def readBatchInfo(PATH):
    CaseSet     = pd.read_csv(PATH.joinpath('CaseSet.csv'),delimiter=';')
    MeasureSets = pd.read_csv(PATH.joinpath('MeasureSet.csv'),delimiter=';')
    AvailableSections = [p for p in PATH.joinpath('BaseData').iterdir() if p.is_file()]
    no_of_sections = len(AvailableSections)
    caselist = []
    for row in CaseSet.iterrows():
        if row[1]['CaseOn'] == 1:

            case_no = row[1]['CaseNumber']
            case_measure_set = pd.DataFrame({'available': MeasureSets[row[1]['MeasureSet']].values})
            for run in range(1, row[1]['Runs']+1):
                sections = np.random.randint(0, no_of_sections, size=row[1]['Sections'])

                #make directory
                casepath = PATH.joinpath('{:02d}'.format(case_no), '{:03d}'.format(run))
                #if the case exists: delete the entire directory
                if casepath.is_dir():
                    rmtree(casepath)
                caselist.append(('{:02d}'.format(case_no), '{:03d}'.format(run)))
                casepath.joinpath('StabilityInner').mkdir(parents= True,exist_ok=True)
                casepath.joinpath('Piping').mkdir(parents= True,exist_ok=True)
                casepath.joinpath('Overflow').mkdir(parents= True,exist_ok=True)
                casepath.joinpath('Toetspeil').mkdir(parents= True,exist_ok=True)
                #copy the section files
                for i in sections:
                    copyfile(AvailableSections[i],casepath.joinpath(AvailableSections[i].name))
                    book = load_workbook(casepath.joinpath(AvailableSections[i].name))
                    writer = pd.ExcelWriter(casepath.joinpath(AvailableSections[i].name),engine='openpyxl')
                    writer.book = book
                    writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
                    case_measure_set.to_excel(writer,'Measures',columns=['available'],index=False,startcol=3)
                    writer.save()
                    #change the measureset
                #then copy all the needed files for
                #StabilityInner
                for p in casepath.iterdir():
                    if p.is_file():
                        data = pd.read_excel(p,sheet_name='General',index_col=0)
                        copyfile(PATH.joinpath('BaseData',data.loc['Overflow']['Value']),casepath.joinpath(data.loc['Overflow']['Value']))
                        copyfile(PATH.joinpath('BaseData',data.loc['StabilityInner']['Value']),casepath.joinpath(data.loc['StabilityInner']['Value']))
                        copyfile(PATH.joinpath('BaseData',data.loc['Piping']['Value']),casepath.joinpath(data.loc['Piping']['Value']))
                        copyfile(PATH.joinpath('BaseData','Toetspeil',data.loc['LoadData']['Value']),casepath.joinpath('Toetspeil',data.loc['LoadData']['Value']))
    return caselist

def setPath(computer):
    raise ValueError('Change the paths!')
    if computer == 'Horizon':
        PATH = Path(r'c:\PythonResults\KPP case')
    elif computer == 'DeltaresLaptop':
        PATH = Path(r'c:\Users\klerk_wj\SurfDrive\01_Projects\01_KPP Bekledingen\KPP case')
    elif computer == 'TULaptop':
        PATH = Path(r'd:\wouterjanklerk\My Documents\01_Projects\01_KPP Bekledingen\KPP case')
    return PATH

def calcRsquared(x,y):
    SStot = np.sum(np.subtract(x, np.mean(x)) ** 2)
    SSreg = np.sum(np.subtract(y, np.mean(x)) ** 2)
    SSres = np.sum(np.subtract(x, y) ** 2)
    Rsq = 1 - np.divide(SSres, SStot)
    return Rsq