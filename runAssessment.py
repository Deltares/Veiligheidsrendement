## This script can redo the assessment for all the individual mechanisms of piping based on the data from pickle files.
## You can also plot the results for the different sections
## Author: Wouter Jan Klerk
## Date: 20180518
import re
import csv
from os import listdir
from os.path import isfile, join
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd

import Mechanisms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import collections

#Again, specify the path
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-4'

# write to file with cPickle/pickle (as binary)
def ld_writeObject(filePath,object):
    f=open(filePath,'wb')
    newData = pickle.dumps(object, 1)
    f.write(newData)
    f.close()

#Function to calculate a safety factor:
def calc_gamma(mechanism,DikeSection):
    Pcs = (DikeSection.Pmax * DikeSection.omega * DikeSection.b) /( DikeSection.a * DikeSection.length)
    betacs = -norm.ppf(Pcs)
    betamax = -norm.ppf(DikeSection.Pmax)
    if mechanism == 'Piping':
        gamma = 1.04*np.exp(0.37*betacs-0.43*betamax)
    elif mechanism == 'Heave':
        gamma = 0.37*np.exp(0.48*betacs-0.3*betamax)
    elif mechanism == 'Uplift':
        gamma = 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
    else:
        print('Mechanism not found')
    return gamma

#Function to calculate the implicated reliability from the safety factor
def calc_beta_implicated(mechanism,SF,DikeSection):
    if mechanism == 'Piping':
        beta = (1 / 0.37) * (np.log(SF / 1.04) + 0.43 * -norm.ppf(DikeSection.Pmax))
    elif mechanism == 'Heave':
        beta = (1 / 0.46) * (np.log(SF / 0.48) + 0.27 * -norm.ppf(DikeSection.Pmax))
    elif mechanism == 'Uplift':
        # print(SF)
        beta = (1 / 0.48) * (np.log(SF / 0.37) + 0.30 * -norm.ppf(DikeSection.Pmax))
    else:
        print('Mechanism not found')
    return beta

#Class for the input of a mechanism. Function only available for piping for now
class MechanismInput:
    def __init__(self,mechanism):
        self.mechanism = mechanism

    def fill_piping_data(self, dict, type):
        if type == 'SemiProb':
            self.inputtype = type
            #Input needed for multiple submechanisms
            self.h           = dict['Input']['Scenario 1']['j 0 '][1]
            self.h_exit      = dict['Input']['Scenario 1']['hp = j 3'][1]
            self.gamma_w     = dict['Input']['Scenario 1']['gw'][1]
            self.r_exit      = (dict['Input']['Scenario 1']['j2'][1] - self.h_exit)/(dict['Input']['Scenario 1']['j 0 '][1]-self.h_exit)
            self.d_cover     = dict['Input']['Scenario 1']['d,pot'][1]
            self.gamma_schem = dict['Input']['Scenario 1']['gb,u'][1]
            #Input parameter specifically for piping
            self.D           = dict['Input']['Scenario 1']['D'][1]
            self.k           = dict['Input']['Scenario 1']['kzand,pip'][1]
            self.L_voorland  = dict['Input']['Scenario 1']['L1:pip'][1]
            self.L_dijk      = dict['Input']['Scenario 1']['L2'][1]
            self.L_berm      = dict['Input']['Scenario 1']['L3'][1]
            self.L           = self.L_berm + self.L_dijk + self.L_voorland
            self.d70         = dict['Input']['Scenario 1']['d70'][1]
            self.d_cover_pip = dict['Input']['Scenario 1']['d,pip'][1]
            self.m_Piping    = 1.0
            self.theta       = 37.
            #Input specific for heave
            self.scherm      = dict['Results']['Scenario 1']['Heave']['kwelscherm aanwezig']

            #Input specific for uplift
            self.gamma_sat   = dict['Input']['Scenario 1']['Gemiddeld volumegewicht:'][1]
            print(self.gamma_sat)
            self.gamma_sat = 18 if self.gamma_sat == 0. else self.gamma_sat
            print(self.gamma_sat)
            # self.d70m        = dict['Input']['Scenario 1']['d70m'] NB: nu and eta are also already defined
        else:
            print('unknown type')

#Class describing an assessment (functions for piping, heave & uplift available)
class Assessment:
    def __init__(self,mechanism,type):
        self.mechanism = mechanism
        self.type = 'SemiProb'

    def Assess(self,DikeSection, MechanismInput):
        #First calculate the SF without gamma for the three submechanisms
        #Piping:
        Z, self.p_dh, self.p_dh_c = Mechanisms.LSF_sellmeijer(MechanismInput)                   #Calculate hydraulic heads
        self.gamma_pip   = calc_gamma('Piping',DikeSection)                                         #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS NOT included here. Which is correct because a scenario approach is taken.
        self.SF_p = (self.p_dh_c/self.gamma_pip)/self.p_dh
        self.assess_p  = 'voldoende' if (self.p_dh_c/self.gamma_pip)/self.p_dh > 1 else 'onvoldoende'
        self.beta_cs_p = calc_beta_implicated('Piping',self.p_dh_c/self.p_dh,DikeSection)     #Calculate the implicated beta_cs

        #Heave:
        Z, self.h_i, self.h_i_c = Mechanisms.LSF_heave(MechanismInput)                                  #Calculate hydraulic heads
        self.gamma_h   = calc_gamma('Heave',DikeSection)                                            #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS included here
        self.SF_h = (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i
        self.assess_h  = 'voldoende' if (self.h_i_c/(MechanismInput.gamma_schem*self.gamma_h))/self.h_i > 1 else 'onvoldoende'
        self.beta_cs_h = calc_beta_implicated('Heave',self.h_i_c/self.h_i,DikeSection)                 #Calculate the implicated beta_cs

        #Uplift
        Z, self.u_dh, self.u_dh_c = Mechanisms.LSF_uplift(MechanismInput)                                  #Calculate hydraulic heads
        self.gamma_u   = calc_gamma('Uplift',DikeSection)                                            #Calculate needed safety factor
        # Check if it is OK, NB: Schematization factor IS included here
        self.SF_u = (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh
        self.assess_u  = 'voldoende' if (self.u_dh_c/(MechanismInput.gamma_schem*self.gamma_u))/self.u_dh > 1 else 'onvoldoende'
        self.beta_cs_u = calc_beta_implicated('Uplift',self.u_dh_c/self.u_dh,DikeSection)                 #Calculate the implicated beta_cs


#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.name = name
        if traject == '16-4':
            self.length = 19480
            self.Pmax = 1. / 10000; self.omega = 0.24; self.a = 0.9; self.b = 300
        elif traject == '16-3':
            self.length = 19899
            self.Pmax = 1. / 10000; self.omega = 0.24; self.a = 0.9; self.b = 300  #NB: klopt a hier?????!!!!

    def fill_from_dict(self, dict):
        #First the general info (to be added: traject info, norm etc)
        self.start = dict['General']['Traject start']
        self.end   = dict['General']['Traject end']
        self.CS    = dict['General']['Cross section']
        self.MHW   = dict['Input']['Scenario 1']['j 0 ']        #TO DO: add a loop over scenarios
        self.PipingIn = MechanismInput('Piping')
        self.PipingIn.fill_piping_data(dict,'SemiProb')

    def doAssessment(self, mechanism, type):
        if mechanism == 'Piping':

            self.PipingAssessment = Assessment(mechanism, type)
            self.PipingAssessment.Assess(self,self.PipingIn)

        else:
            print('Mechanism not known')


#Make a list of the files used as input
onlyfiles = [f for f in listdir(pad) if isfile(join(pad,f))]

#This writes a full xlsx with all section info
def writedataforNelle():
    y = pd.DataFrame(flatten(allsections[sectionnames[0]]), index=['Beschrijving', 'remove'])
    y = y.T
    y = y.drop(columns=['remove'])
    for i in sectionnames:
        data = flatten(allsections[i])
        x = pd.DataFrame(data, index=['remove', i])
        x = x.T
        x = x.drop(columns=['remove'])
        y = y.join(x, how='inner', rsuffix=i)
    writer = pd.ExcelWriter('D:/wouterjanklerk/Desktop/allvalues.xlsx')
    y.to_excel(writer, 'Blad1')
    writer.save()

#Helper function for full write script
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#Plot function for the assessment results
def plotCS(Sections,pad, type,extra_variable):
    # # This function plots the reliability index for all sections for all submechanisms.
    beta_cs_p = []; beta_cs_h = []; beta_cs_u = []; beta_cs_max = []; sec_name = []
    for i in range(0,len(Sections)):
        sec_name.append(Sections[i].name)
        beta_cs_p.append(Sections[i].PipingAssessment.beta_cs_p)
        beta_cs_h.append(Sections[i].PipingAssessment.beta_cs_h)
        beta_cs_u.append(Sections[i].PipingAssessment.beta_cs_u)
        beta_cs_max.append(max(Sections[i].PipingAssessment.beta_cs_p,Sections[i].PipingAssessment.beta_cs_u,Sections[i].PipingAssessment.beta_cs_h))


    if type == 'Beta':
        plt.plot(range(0,len(Sections)), beta_cs_p, 'or', alpha=0.4, label = 'Piping')
        plt.plot(range(0,len(Sections)), beta_cs_u, 'ob', alpha=0.4, label = 'Uplift')
        plt.plot(range(0,len(Sections)), beta_cs_h, 'og', alpha=0.4, label = 'Heave')
        plt.plot(range(0,len(Sections)), beta_cs_max, 'k', label = 'highest')
        plt.xticks(range(0,len(Sections)), sec_name, rotation='vertical')

        plt.ylabel(r'Cross sectional reliability index $\beta$')
        plt.ylim(ymin=0)

    elif type == 'Probability':
        plt.plot(range(0,len(Sections)), norm.cdf(-np.array(beta_cs_p)), 'or', alpha=0.4, label='Piping')
        plt.plot(range(0,len(Sections)), norm.cdf(-np.array(beta_cs_u)), 'ob', alpha=0.4, label='Uplift')
        plt.plot(range(0,len(Sections)), norm.cdf(-np.array(beta_cs_h)), 'og', alpha=0.4, label='Heave')
        plt.plot(range(0,len(Sections)), norm.cdf(-np.array(beta_cs_max)), 'k', label='highest')
        plt.xticks(range(0,len(Sections)), sec_name, rotation='vertical')
        plt.ylabel(r'Cross sectional failure probability $P_f$')
        plt.legend(loc='upper right')
        # plt.ylim(ymin=0)
        plt.yscale('log')

        # plt.savefig(pad + r'\output\FailureProbability.png')

    if extra_variable == 'Cover layer':
        d_c = []; sec_name = [];
        for i in range(0, len(Sections)):
            sec_name.append(Sections[i].name)
            d_c.append(Sections[i].PipingIn.d_cover_pip)
        ax2 = plt.twinx()
        color = 'tab:blue'
        ax2.plot(range(0,len(Sections)),np.array(d_c), 'tab:blue', alpha=0.2, label='Deklaag')
        ax2.set_ylabel('Cover layer in m')
        ax2.tick_params(axis='y', labelcolor=color)
    elif extra_variable == 'Seepage length':
        L = []; sec_name = [];
        for i in range(0, len(Sections)):
            sec_name.append(Sections[i].name)
            L.append(Sections[i].PipingIn.L_berm+Sections[i].PipingIn.L_voorland+Sections[i].PipingIn.L_dijk)
        ax2 = plt.twinx()
        color = 'tab:blue'
        ax2.plot(range(0,len(Sections)),np.array(L), 'tab:blue', alpha=0.2, label='Leklengte')
        ax2.set_ylabel('Seepage length in m')
        ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()
#Used to read a pickle file which can contain dictionaries with input
def ld_readDicts(filePath):
    # This script can load the pickle file so you have a nice dictionary
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

#Get a list of results (beta or safety factors) from the Class structure of a section
def extractResult(Sections,type):
    if type == 'SF':
        X_p = []; X_u = []; X_h = [];
        for i in range(0, len(Sections)):
            X_p.append(Sections[i].PipingAssessment.SF_p)
            X_u.append(Sections[i].PipingAssessment.SF_u)
            X_h.append(Sections[i].PipingAssessment.SF_h)
    elif type == 'beta':
        X_p = []; X_u = []; X_h = [];
        for i in range(0, len(Sections)):
            X_p.append(Sections[i].PipingAssessment.beta_cs_p)
            X_u.append(Sections[i].PipingAssessment.beta_cs_u)
            X_h.append(Sections[i].PipingAssessment.beta_cs_h)
    return X_p, X_u, X_h

#Extract the safety fcator from an existing assessment by WSRL. Used for verification of the code.
def extractSF_WSRL(allsections):
    SF_p = [];     SF_u = [];     SF_h = [];
    for i in sectionnames:
        #Piping
        dH = allsections[i]['Results']['Scenario 1']['Piping']['optredend verval'] #0.3*d is already in her
        dHc = 0 if allsections[i]['Results']['Scenario 1']['Piping']['kritiek verval'] == "#DIV/0!" else float(allsections[i]['Results']['Scenario 1']['Piping']['kritiek verval'])
        gamma_p = allsections[i]['Input']['Safety']['gamma_pip'][1]
        #correct for wrong safety factors and incorrectly accounted for schematization factor:
        dHc = dHc*gamma_p*1.3

        SF_p.append((dHc/gamma_p)/dH)
        # SF_p.append((dHc)/dH)

        #Heave
        i_c         = float(allsections[i]['Results']['Scenario 1']['Heave']['toelaatbaar verhang'])
        ih          = 0 if allsections[i]['Results']['Scenario 1']['Heave']['optredend heave gradient'] == "#DIV/0!" else float(allsections[i]['Results']['Scenario 1']['Heave']['optredend heave gradient'])
        gamma_schem = float(allsections[i]['Results']['Scenario 1']['Heave']['schematiseringsfactor'])
        gamma_h     = float(allsections[i]['Results']['Scenario 1']['Heave']['veiligheidsfactor'])
        SF_h.append((i_c/(gamma_schem*gamma_h))/ih) if ih != 0 else SF_h.append(0)
        # print(i)

        #Uplift
        sfu = 100 if allsections[i]['Results']['Scenario 1']['Uplift']['u.c.'][1]=="#DIV/0!" else float(allsections[i]['Results']['Scenario 1']['Uplift']['u.c.'][1])
        SF_u.append(1/(sfu))
    return SF_p, SF_u, SF_h


## HERE THE ACTUAL SCRIPT STARTS

#First put all the data in 1 big dictionary for all sections in the traject
allsections = {}
sectionnames = []
# # return dict data to new dict
for i in onlyfiles:
    allsections[i.split(' ')[1][:-5]] = ld_readDicts(pad + '\\output\\' + i.split(' ')[1][:-5] + '.dta')
    sectionnames.append(i.split(' ')[1][:-5])

# with open(pad+ r"\output\assessment.csv",'w') as resultFile:
#     wr = csv.writer(resultFile, dialect='excel',lineterminator='\r',delimiter=';')
#     wr.writerows(data)
#
#
Sections = []
for i in sectionnames:
    Sections.append(DikeSection(i,'16-4'))
    Sections[-1].fill_from_dict(allsections[i])
    Sections[-1].doAssessment('Piping','SemiProb')
# beta_cs_p, beta_cs_u, beta_cs_h = extractResult(Sections,'beta')
ld_writeObject(pad + '\\output\\AllSections.dta', Sections)
plotCS(Sections,pad,'Beta','Seepage length')
plotCS(Sections,pad,'Beta','Cover layer')
# plotCS(Sections,pad,'Beta','None')




# # SF_p_WSRL, SF_u_WSRL, SF_h_WSRL = extractSF_WSRL(allsections)
# plt.plot(SF_p,SF_p_WSRL,'xr')
# plt.plot(np.linspace(0,1),np.linspace(0,1),'k--')
# plt.plot(SF_u,SF_u_WSRL,'xb')
# plt.plot(SF_h,SF_h_WSRL,'xg')
# plt.xlabel('self')
# plt.ylabel('WSRL')
# plt.ylim((0,1))
# plt.xlim((0,1))
# plt.show()
# print()
# allsections = ld_readDicts('C:/Users/Lee/Desktop/test2.dta')
