try:
    import cPickle as pickle
except:
    import pickle
import shelve
import copy
import pandas as pd
import matplotlib.pyplot as plt
import openturns as ot
from pandas import DataFrame
import numpy as np
import os

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
        name = TestCaseSolutions[section].MeasureTable.loc[TestCaseSolutions[section].MeasureTable['ID'] == id]['Name'].values
        TestCaseStrategy.TakenMeasures.at[i, 'name'] = name
    return TestCaseStrategy


def getMeasureTable(AllSolutions):
    OverallMeasureTable = pd.DataFrame([], columns=['ID', 'Name'])

    for i in AllSolutions:
        OverallMeasureTable = pd.concat([OverallMeasureTable, AllSolutions[i].MeasureTable])

    OverallMeasureTable = OverallMeasureTable.drop_duplicates(subset='ID')
    return OverallMeasureTable




def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#this is a more generic function to read and write data from and to a shelve. But it is not implemented fully:
# TODO implement DataAtShelve instead of (un)commenting snippets of code
def DataAtShelve(dir, name, objects = None, mode = 'write'):
    if mode == 'write':
        #make shelf
        my_shelf = shelve.open(dir + '\\' + name, 'n')
        for i in objects.keys():
            my_shelf[i] = objects[i]
        my_shelf.close()
    elif mode == 'read':
        # open shelf
        my_shelf = shelve.open(dir + '\\' + name)
        keys = []
        for key in my_shelf:
            locals()[key]=my_shelf[key]
            keys.append(key)
        my_shelf.close()
        if len(keys) == 1:
            return locals()[keys[0]]