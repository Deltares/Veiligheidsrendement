## This script can redo the assessment for all the individual mechanisms of piping based on the data from pickle files.
## You can also plot the results for the different sections
## Author: Wouter Jan Klerk
## Date: 20180518
import re
import csv
from os import listdir
from os.path import isfile, join
from HelperFunctions import ld_writeObject, ld_readObject, flatten
import pandas as pd
from DikeClasses import DikeSection
import matplotlib.pyplot as plt
import numpy as np
import operator
from scipy.stats import norm
import collections

#Specify the path
traject = '16-4'
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data' + '\\' + traject + '\\output'

#Make a list of the files used as input
onlyfiles = [f for f in listdir(pad) if isfile(join(pad,f))]

#This writes a full xlsx with all section info
def writetoExcel():
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

#Plot function for the assessment results
def plotCS(Sections,mechanism, type,extra_variable,mode='show',savepath=None):
    Sections.sort(key=operator.attrgetter('name'))
    if mechanism == 'Piping':
        # # This function plots the reliability index for all sections for all piping submechanisms.
        beta_cs_p = []; beta_cs_h = []; beta_cs_u = []; beta_cs_max = []; sec_name = []
        for i in range(0,len(Sections)):
            sec_name.append(Sections[i].name)
            beta_cs_p.append(Sections[i].Reliability.Piping.beta_cs_p)
            beta_cs_h.append(Sections[i].Reliability.Piping.beta_cs_h)
            beta_cs_u.append(Sections[i].Reliability.Piping.beta_cs_u)
            beta_cs_max.append(max(Sections[i].Reliability.Piping.beta_cs_p,Sections[i].Reliability.Piping.beta_cs_u,Sections[i].Reliability.Piping.beta_cs_h))


        if type == 'Beta':
            plt.plot(range(0,len(Sections)), beta_cs_p, 'or', alpha=0.4, label = 'Piping')
            plt.plot(range(0,len(Sections)), beta_cs_u, 'ob', alpha=0.4, label = 'Uplift')
            plt.plot(range(0,len(Sections)), beta_cs_h, 'og', alpha=0.4, label = 'Heave')
            plt.plot(range(0,len(Sections)), beta_cs_max, 'k', label = 'highest')
            plt.xticks(range(0,len(Sections)), sec_name, rotation='vertical')
            plt.legend(loc='upper right')
            plt.plot([0, len(Sections)], [4.93, 4.93])

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

            plt.savefig(pad + r'\output\BetaTraject' + traject + '.pdf')

        if extra_variable == 'Cover layer':
            d_c = []; sec_name = [];
            for i in range(0, len(Sections)):
                sec_name.append(Sections[i].name)
                d_c.append(Sections[i].Reliability.Piping.Input.d_cover_pip)
            ax2 = plt.twinx()
            color = 'tab:blue'
            ax2.plot(range(0,len(Sections)),np.array(d_c), 'tab:blue', alpha=0.2, label='Deklaag')
            ax2.set_ylabel('Cover layer in m')
            ax2.tick_params(axis='y', labelcolor=color)
        elif extra_variable == 'Seepage length':
            L = []; sec_name = [];
            for i in range(0, len(Sections)):
                sec_name.append(Sections[i].name)
                L.append(Sections[i].Reliability.Piping.Input.L_berm+Sections[i].Reliability.Piping.Input.L_voorland+Sections[i].Reliability.Piping.Input.L_dijk)
            ax2 = plt.twinx()
            color = 'tab:blue'
            ax2.plot(range(0,len(Sections)),np.array(L), 'tab:blue', alpha=0.2, label='Leklengte')
            ax2.set_ylabel('Seepage length in m')
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            plt.plot()
    plt.tight_layout()
    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig(savepath + r'\output' + '\\' + type + '_Traject_' + traject + '_' + extra_variable + '.pdf')
        plt.close()

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
#For this we use the pickle files that were generated using readQuickScan.py
allsections = {}
sectionnames = []
onlyfiles = [f for f in listdir(pad) if isfile(join(pad,f))]
## read the input
for i in onlyfiles:
    allsections[i[:-4]] = ld_readObject(pad + '\\' + i)
    sectionnames.append(i[:-4])

#make Sections, a list of class objects
Sections = []
for i in sectionnames:
    Sections.append(DikeSection(i,traject))
    Sections[-1].fill_from_dict(allsections[i])
    Sections[-1].doAssessment('Piping','SemiProb')

#Write the results
# ld_writeObject(pad + '\\input\\AllSections.dta', Sections)

#Plot the results
plotCS(Sections,'Piping','Beta','Cover layer')
plotCS(Sections,'Piping','Beta','Seepage length')
