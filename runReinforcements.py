
import matplotlib.pyplot as plt
import csv
import copy
import pandas as pd
import numpy as np
from scipy.stats import norm
try:
    import cPickle as pickle
except:
    import pickle

# read file decoding with cPickle/pickle (as binary)
def ld_readObject(filePath):
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

def calcTrajectProb(P_sections):
    P_traject = sum(P_sections)
    return P_traject

def calcAnnuityFactor(T,r):
    A = ((1 - (1 + r) ** -T) / r)
    return A
def calcLCC(measure, T, r, section):
    A = calcAnnuityFactor(T,r)
    LCC = A*measure['C_var']*section['Length']+measure['C_fix']
    return LCC.values[0]

def calcLCR(measure, T, r, D, P_sections,section):
    A = calcAnnuityFactor(T,r)
    #recalculate the probabilities for a measure:
    if measure['P_type'] == 'factor':
        P_sections.loc[section] = P_sections.loc[section]* measure['P']
    elif measure['P_type'] == 'fixed':
        P_sections.loc[section] = measure['P']
    P_traject = calcTrajectProb(P_sections)
    LCR = P_traject * A * D
    return LCR, P_traject

def MeasureCostBenefits(sections, measures, general):
    P_traject = calcTrajectProb(sections['Pvak'])
    # Psections = copy.deepcopy(sections['Pvak'])
    A = calcAnnuityFactor(general['T'],general['r'])
    LCR = P_traject*general['D']*A
    cols = sections.index.tolist()
    MeasureCosts = pd.DataFrame(columns=cols, index = range(measures.shape[0]))
    MeasureRisk  = pd.DataFrame(columns=cols, index = range(measures.shape[0]))
    MeasureP     = pd.DataFrame(columns=cols, index = range(measures.shape[0]))
    for i in sections.index:
        Psections = copy.deepcopy(sections['Pvak'])
        for j in range(0,len(measures)):
            MeasureCosts[i][j] = calcLCC(measures.iloc[j,:], general['T'], general['r'], sections.loc[sections.index == i])
            MeasureRisk[i][j], MeasureP[i][j] = calcLCR(measures.iloc[j,:], general['T'], general['r'], general['D'], Psections,i)
             # = LCR_meas - LCR
    MeasureCosts.index = measures.index
    MeasureRisk.index = measures.index
    deltaTotalCost = MeasureCosts + MeasureRisk - LCR
    deltaTotalCost.index = measures.index

    BCratio = (LCR-MeasureRisk)/MeasureCosts
    return deltaTotalCost, BCratio, MeasureCosts


def writetoDataFrame(DF,index,columns,data):
    count = 0
    for i in columns:
        DF[i][index] = data[count]
        count = count+1
    return DF

def takeMeasure(measure,section,measures,sections):
    # measures = measures.set_index('naam')
    if measures.loc[measure,'P_type'] == 'fixed':
        sections.loc[section,'Pvak'] = measures.loc[measure,'P']
    elif measures.loc[measure,'P_type'] == 'factor':
        sections.loc[section,'Pvak'] = sections.loc[section,'Pvak'] * measures.loc[measure,'P']
    return sections, measures

def selectMeasure(sections,measures,general,TCmeasures,BCratio,Measures):
    mode = 'BC'
    if mode == 'TC':
        sec = [];     TC = 1e15
        #this code can be much shorter I think!!!
        for i in sections.index:
            if min(TCmeasures.loc[:,i])<TC:
                TC = min(TCmeasures.loc[:,i])
                sec = i
            # else:
            #     #do nothing
        #section selected. Now select the proper intervention
        interv = []
        TC = []
        for i in measures.index:
            if min(TCmeasures.loc[:,sec])==TCmeasures.loc[i,sec]:
                measure = i
                location = sec
                #optimum found: section is sec and measure is i
    elif mode == 'BC':
        sec = [];     BC = -1000
        #this code can be much shorter I think!!!
        for i in sections.index:
            if max(BCratio.loc[:,i])>BC:
                BC = max(BCratio.loc[:,i])
                sec = i
            # else:
            #     #do nothing
        #section selected. Now select the proper intervention
        interv = []; BC = []
        for i in measures.index:
            if max(BCratio.loc[:,sec])==BCratio.loc[i,sec]:
                measure = i
                location = sec
                #optimum found: section is sec and measure is i
    return measure, location


def makePlanning(sections, measures, general):
    #Calculate base values for probability and risk
    P_traject = np.sum(norm.cdf(-sections['beta vak']))
    A = calcAnnuityFactor(general['T'],general['r'])
    LCR = A*P_traject*general['D']
    sections['Pvak'] = norm.cdf(-sections['beta vak'])

    #Initialize lists and dataframes that are needed
    P_sections =[]; Costs = []; Measures = []; Measures.append(['Initial', '', 0, P_traject]); section_probs =[];
    # P_sections = pd.DataFrame(columns=sections['Section Name'], index=range(general['T']))
    # cols = ['naam'] + sections['Section Name'].values.tolist()
    P_traject_list = [];
    # P_sections = writetoDataFrame(P_sections, 0, sections['Section Name'], sections['Pvak'])
    for i in range(0,4*len(sections)):
        dTotalCost, BCratio, Costs = MeasureCostBenefits(sections,measures,general)
        measure, section = selectMeasure(data,measures,general,dTotalCost,BCratio,Measures)
        sections, measures = takeMeasure(measure, section, measures, sections)
        P_traject_list.append(calcTrajectProb(sections['Pvak']))
        Measures.append([section, measure, Costs.loc[measure,section], P_traject_list[-1]])
        #record the measure
    return Measures

pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\DataNelle\SommenQuickScan\16-4\output'
a = 0.9
b = 300
data = pd.read_csv(pad+ r"\assessment.csv",delimiter=';')
data['Length'] = data['km end'] - data['km start']
data['beta vak'] = -norm.ppf(norm.cdf(-data['beta_cs'])*(a*1000*data['Length']/b))
data = data.set_index('Section Name')
general = {}
general['Pmax'] = 1/10000
general['r'] = 0.03
general['T'] = 50
general['D'] = 23e9

P_traject = np.sum(norm.cdf(-data['beta vak']))

measures = pd.read_csv(pad+ r"\measures.csv",delimiter=';')
measures = measures.set_index('naam')



data_original = copy.deepcopy(data)
measures_original = copy.deepcopy(data)
general_original = copy.deepcopy(data)
Measures = makePlanning(data,measures,general)
print()



with open("interventions.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel',lineterminator='\r',delimiter=';')
    wr.writerows(Measures)







