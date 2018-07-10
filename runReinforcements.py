
import matplotlib.pyplot as plt
import csv
import copy
import pandas as pd
import numpy as np
import numpy.ma as ma
from HelperFunctions import ld_readObject
from scipy.stats import norm
from ProbabilisticFunctions import calc_traject_prob
import operator
class Traject:
    def __init__(self, traject, Sections):
        self.name = traject
        if traject == '16-4':
            self.length = 19480
            self.Pmax = 1. / 10000; #self.omegaPiping = 0.24; self.a = 0.9; self.b = 300
        elif traject == '16-3':
            self.length = 19899
            self.Pmax = 1. / 10000; #self.omegaPiping = 0.24; self.a = 0.9; self.b = 300  #NB: klopt a hier?????!!!!
        self.Sections = Sections
        self.Sections.sort(key=operator.attrgetter('name'))

    def determineSectionLengths(self):
        dijkpaal_dist = 200
        DV = []
        self.Sections.sort(key=operator.attrgetter('CS'))
        for i in range(0, len(self.Sections)):
            startkm = float(self.Sections[i].start[2:5] + '.' + format(int(self.Sections[i].start[6:]), '03d')) if len(self.Sections[i].start) > 5 else float(self.Sections[i].start[2:5])
            endkm   = float(self.Sections[i].end[2:5] + '.' + format(int(self.Sections[i].end[6:]), '03d'))     if len(self.Sections[i].end)   > 5 else float(self.Sections[i].end[2:5])
            if self.Sections[i].name[0:4] == DV:
                #Functie klopt niet helemaal: uiteindelijk moet dit gewoon in GIS worden aangegeven.
               #adapt upper bound and length of previous section
                print(self.Sections[i-1].CS)
                CS_neighbour = float(self.Sections[i-1].CS[2:5] + '.' + format(int(self.Sections[i-1].CS[6:][0:3]), '03d'))
                CS           = float(self.Sections[i].CS[2:5] + '.' + format(int(self.Sections[i].CS[6:][0:3]), '03d'))
                newbound     = np.mean([CS,CS_neighbour])
                #Recalculate previous:
                start_prev = float(self.Sections[i-1].start[2:5] + '.' + self.Sections[i-1].start[6:])
                self.Sections[i-1].length = (newbound-start_prev)*dijkpaal_dist
                newupper = self.Sections[i].CS[0:2] + format(int(np.floor(newbound)), '03d') + '+' + format(int((newbound - np.floor(newbound)) * 100), '03d')
                newupper = self.Sections[i].CS[0:2] + format(int(np.floor(newbound)),'03d') + '+' + format(int((newbound-np.floor(newbound))*100),'03d')
                self.Sections[i-1].end    = newupper
                #Write new length and boundaries
                self.Sections[i].start = newupper
                self.Sections[i].length= (endkm-newbound)*dijkpaal_dist

            else:
                DV = self.Sections[i].name[0:4]
                self.Sections[i].length = (endkm-startkm)*dijkpaal_dist

        self.Sections.sort(key=operator.attrgetter('name'))
            #Testcode
            # lensum = 0
            # for i in range(0,len(self.Sections)):
            #     print(self.Sections[i].name + ' ' + self.Sections[i].start + ' ' + self.Sections[i].CS  + ' ' + self.Sections[i].end + ' ' + self.Sections[i].length)
            #     lensum = lensum + self.Sections[i].length
            # print(lensum)

    def calcBetaTCS(self,mechanism):
        if mechanism == 'Piping':
            for i in range(0,len(self.Sections)):
                Peis = self.Sections[i].TrajectInfo['omegaPiping']* self.Sections[i].TrajectInfo['Pmax']
                N = self.Sections[i].TrajectInfo['aPiping'] * self.Sections[i].TrajectInfo['TrajectLength'] / self.Sections[i].TrajectInfo['bPiping']
                self.Sections[i].BetaCSPiping = -norm.ppf(Peis/N)
    def calcPCS(self,mechanism):
        if mechanism == 'Piping':
            for i in range(0,len(self.Sections)):
                self.Sections[i].PipingAssessment.Pf = norm.cdf(-(max((self.Sections[i].PipingAssessment.beta_cs_h,self.Sections[i].PipingAssessment.beta_cs_p,self.Sections[i].PipingAssessment.beta_cs_u))))
    def setReinfVars(self,discountrate,planningperiod,flooddamage):
        self.DiscountRate = discountrate
        self.PlanningPeriod = planningperiod
        self.FloodDamage = flooddamage

    def calcPcurrent(self):
        self.Pcurrent = calc_traject_prob(self.Sections, 'Piping')

class Measures:
    def __init__(self):
        self.MeasureList = []

    def addMeasuresfromCSV(self, pad):
        measures = pd.read_csv(pad + r"\measures.csv", delimiter=';')
        measures = measures.set_index('naam')
        for i in range(0,len(measures)):
            self.MeasureList = measures

def calcTrajectProb(P_sections):
    P_traject = sum(P_sections)
    return P_traject

def calcAnnuityFactor(T,r):
    A = ((1 - (1 + r) ** -T) / r)
    return A
def calcLCC(measure, T, r, section):
    A = calcAnnuityFactor(T,r)
    LCC = A*measure['C_var']*(section.length/1000.)+measure['C_fix']
    return LCC

def calcLCR(measure, T, r, D, P_sections,section):
    A = calcAnnuityFactor(T,r)
    #recalculate the probabilities for a measure:
    if measure['P_type'] == 'factor':
        P_sections[section] = P_sections[section]* measure['P']
    elif measure['P_type'] == 'fixed':
        P_sections[section] = measure['P']
    P_traject = calc_traject_prob(P_sections, 'Piping')
    LCR = P_traject * A * D
    return LCR, P_traject

# def MeasureCostBenefits(sections, measures, general):
def MeasureCostBenefits(Traject,Measures):
    Traject.calcPcurrent()
    A = calcAnnuityFactor(Traject.PlanningPeriod,Traject.DiscountRate)
    LCR = A*Traject.FloodDamage*Traject.Pcurrent

    MeasureCosts = pd.DataFrame(columns = range(len(Traject.Sections)), index = range(Measures.MeasureList.shape[0]))
    MeasureRisk  = pd.DataFrame(columns = range(len(Traject.Sections)), index = range(Measures.MeasureList.shape[0]))
    MeasureP     = pd.DataFrame(columns = range(len(Traject.Sections)), index = range(Measures.MeasureList.shape[0]))
    P_sections = []
    P_sections = [Traject.Sections[i].PipingAssessment.Pf for i in range(0,len(Traject.Sections))]
    for i in range(0,len(Traject.Sections)):
        for j in range(0,len(Measures.MeasureList)):
            MeasureCosts[i][j] = calcLCC(Measures.MeasureList.iloc[j,:],Traject.PlanningPeriod,Traject.DiscountRate,Traject.Sections[i])
            Psections = copy.deepcopy(P_sections)
            MeasureRisk[i][j], MeasureP[i][j] = calcLCR(Measures.MeasureList.iloc[j,:], Traject.PlanningPeriod,Traject.DiscountRate, Traject.FloodDamage, Psections,i)
             # = LCR_meas - LCR
    MeasureCosts.index = Measures.MeasureList.index
    MeasureRisk.index = Measures.MeasureList.index
    deltaTotalCost = MeasureCosts + MeasureRisk - LCR
    deltaTotalCost.index = Measures.MeasureList.index

    BCratio = (LCR-MeasureRisk)/MeasureCosts
    return deltaTotalCost, BCratio, MeasureCosts

def writetoDataFrame(DF,index,columns,data):
    count = 0
    for i in columns:
        DF[i][index] = data[count]
        count = count+1
    return DF

def takeMeasure(idx,measures,traject):
    if measures.MeasureList.P_type[idx[0]] == 'fixed':
        traject.Sections[idx[1]].PipingAssessment.__clearvalues__()
        traject.Sections[idx[1]].PipingAssessment.Pf = measures.MeasureList.P[idx[0]]
        traject.Sections[idx[1]].ReinfDone = 1 #MAKE MORE ROBUST
        print('do this')
    elif measures.MeasureList.P_type[idx[0]] == 'factor':
        P_f = traject.Sections[idx[1]].PipingAssessment.Pf
        traject.Sections[idx[1]].PipingAssessment.__clearvalues__()
        traject.Sections[idx[1]].PipingAssessment.Pf = measures.MeasureList.P[idx[0]]*P_f
        traject.Sections[idx[1]].MaintDone = 1
    return traject
def makeMask(Traject,shape,types):
    #Makes a mask of all already executed reinforcement or maintenance types
    masker = np.zeros(shape)
    for i in range(0,len(types)):
        for j in range(0,len(Traject.Sections)):
            if types[i] == 'Reinf':
                masker[i,j] = Traject.Sections[j].ReinfDone
            elif types[i] == 'Maint':
                masker[i,j] = Traject.Sections[j].MaintDone
    return masker
def selectMeasure(DecisionVariable,Traject,type,Measures):
    masker = makeMask(Traject,np.shape(DecisionVariable),Measures.MeasureList['type'].values)
    #First we mask all values that are not interesting. Check later if the mask functionality can be used.

    DecisionVariable = ma.masked_array(DecisionVariable.values, mask=masker)
#Function selects a measure based on the optimal value for the decision variable. Returns a tuple with indices for measure and section
    if type == 'BC':
        idx0 = np.unravel_index(DecisionVariable.argmax(fill_value = 0),DecisionVariable.shape)
        # Pvak = Traject.Sections[idx0[1]].PipingAssessment.Pf
        #
        # for i in range(0,len(DecisionVariable)):
        #
        # zoek hogere BC in kolom
        # als gevonden, vergelijk de kansreductie
        # Pak degene waar BC positief is en de kansreductie maximaal
    elif type == 'TC':
        idx0 = np.unravel_index(DecisionVariable.argmin(fill_value = 1e99),DecisionVariable.shape)
    else:
        print('type is not known')
        idx0 = None
    return idx0

def makePlanning(Traject, Measures):
    # A = calcAnnuityFactor(Traject.PlanningPeriod,Traject.DiscountRate)
    # LCR = A*Traject.Pcurrent*Traject.FloodDamage
    PTrajectList = []; InvestmentScheme = []
    InvestmentScheme.append(['Initial','',0.,Traject.Pcurrent])
    for i in range(0,len(Traject.Sections)):
        Traject.Sections[i].ReinfDone = 0
        Traject.Sections[i].MaintDone = 0
    for i in range(0,2*len(Traject.Sections)):
        dTotalCost, BCratio, Costs = MeasureCostBenefits(Traject, Measures)
        # measure_idx = selectMeasure(BCratio,Traject,'BC',Measures)
        measure_idx = selectMeasure(dTotalCost,Traject,'TC',Measures)

        takeMeasure(measure_idx,Traject1_Measures,Traject1)
        Traject.calcPcurrent()
        InvestmentScheme.append([Measures.MeasureList.index[measure_idx[0]], Traject.Sections[measure_idx[1]].name, Costs.iloc[measure_idx], Traject.Pcurrent])
    return InvestmentScheme

def plotCostsVSProbability(Investments):
    totalcosts = np.cumsum(Investments.iloc[:,2])
    plt.plot(totalcosts,Investments.iloc[:,3])
    plt.yscale('log')
    plt.show()
    print()
#Initialize the section information
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-4\input'
Sections = ld_readObject(pad + '\\AllSections.dta')
Traject1 = Traject('16-4',Sections)
Traject1.determineSectionLengths()
Traject1.calcBetaTCS('Piping')
Traject1.calcPCS('Piping')
Traject1.setReinfVars(0.03,50,23e9)
Traject1.calcPcurrent()

#Initialize the measures
measure_pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-4\input'
Traject1_Measures = Measures()
Traject1_Measures.addMeasuresfromCSV(measure_pad)

Traject1_original = copy.deepcopy(Traject1)
Investments = makePlanning(Traject1,Traject1_Measures)

# with open("interventionsTC.csv",'w') as resultFile:
#     wr = csv.writer(resultFile, dialect='excel',lineterminator='\r',delimiter=';')
#     wr.writerows(Investments)

plotCostsVSProbability(pd.DataFrame(Investments))






