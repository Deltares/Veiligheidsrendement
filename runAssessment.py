## This script can redo the assessment for all the individual mechanisms of piping based on the data from pickle files.
## You can also plot the results for the different sections
## Author: Wouter Jan Klerk
## Date: 20180501
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
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\DataNelle\SommenQuickScan\16-4'

#And make a list of the files
onlyfiles = [f for f in listdir(pad) if isfile(join(pad,f))]
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


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def plotBetaCS(allsections,pad):
    # This function plots the reliability index for all sections for all submechanisms.
    SF_piping = [];
    beta_cs_p = []
    SF_heave = [];
    beta_cs_h = []
    SF_uplift = [];
    beta_cs_u = []
    beta_cs_max = []
    sec_num = [];
    for i in sectionnames:
        sec_num.append(i)
        SF_piping.append(allsections[i]['Assessment']['Piping']['Scenario 1']['SF'])
        beta_cs_p.append(allsections[i]['Assessment']['Piping']['Scenario 1']['beta_cross'])
        SF_heave.append(allsections[i]['Assessment']['Heave']['Scenario 1']['SF'])
        beta_cs_h.append(allsections[i]['Assessment']['Heave']['Scenario 1']['beta_cross'])
        SF_uplift.append(allsections[i]['Assessment']['Uplift']['Scenario 1']['SF'])
        beta_cs_u.append(allsections[i]['Assessment']['Uplift']['Scenario 1']['beta_cross'])
        beta_cs_max.append(max(allsections[i]['Assessment']['Piping']['Scenario 1']['beta_cross'],
                               allsections[i]['Assessment']['Heave']['Scenario 1']['beta_cross'],
                               allsections[i]['Assessment']['Uplift']['Scenario 1']['beta_cross']))

    plt.plot(sec_num, beta_cs_p, 'or', alpha=0.4)
    plt.plot(sec_num, beta_cs_u, 'ob', alpha=0.4)
    plt.plot(sec_num, beta_cs_h, 'og', alpha=0.4)
    plt.plot(sec_num, beta_cs_max, 'k')
    plt.xticks(sec_num, sectionnames, rotation='vertical')
    plt.ylabel(r'Cross sectional reliability index $\beta$')
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()

def plotPfCS(allsections,pad):
    # Plot the failure probability for each section
    SF_piping = [];
    beta_cs_p = []
    SF_heave = [];
    beta_cs_h = []
    SF_uplift = [];
    beta_cs_u = []
    beta_cs_max = []
    sec_num = [];
    for i in sectionnames:
        sec_num.append(i)
        SF_piping.append(allsections[i]['Assessment']['Piping']['Scenario 1']['SF'])
        beta_cs_p.append(allsections[i]['Assessment']['Piping']['Scenario 1']['beta_cross'])
        SF_heave.append(allsections[i]['Assessment']['Heave']['Scenario 1']['SF'])
        beta_cs_h.append(allsections[i]['Assessment']['Heave']['Scenario 1']['beta_cross'])
        SF_uplift.append(allsections[i]['Assessment']['Uplift']['Scenario 1']['SF'])
        beta_cs_u.append(allsections[i]['Assessment']['Uplift']['Scenario 1']['beta_cross'])
        beta_cs_max.append(max(allsections[i]['Assessment']['Piping']['Scenario 1']['beta_cross'],
                               allsections[i]['Assessment']['Heave']['Scenario 1']['beta_cross'],
                               allsections[i]['Assessment']['Uplift']['Scenario 1']['beta_cross']))

    plt.plot(sec_num, norm.cdf(-np.array(beta_cs_p)), 'or', alpha=0.4, label = 'Piping')
    plt.plot(sec_num, norm.cdf(-np.array(beta_cs_u)), 'ob', alpha=0.4, label = 'Uplift')
    plt.plot(sec_num, norm.cdf(-np.array(beta_cs_h)), 'og', alpha=0.4, label = 'Heave')
    plt.plot(sec_num, norm.cdf(-np.array(beta_cs_max)), 'k',label = 'highest')
    plt.xticks(sec_num, sectionnames, rotation='vertical')
    plt.ylabel(r'Cross sectional failure probability $P_f$')
    plt.legend(loc='upper right')
    # plt.ylim(ymin=0)
    plt.yscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(pad + r'\output\FailureProbability.png')
def ld_readDicts(filePath):
    # This script can load the pickle file so you have a nice dictionary
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

def heaveAssessment(QS_dictionary,scenario):
    # This is a function for the heave assessment
    #h=water level; h_exit = water level at exit point; r_exit = damping factor
    h           = QS_dictionary['Input']['Scenario ' + str(scenario)]['j 0 '][1]
    h_exit      = QS_dictionary['Input']['Scenario ' + str(scenario)]['hp = j 3'][1]
    r_exit      = (QS_dictionary['Input']['Scenario ' + str(scenario)]['j2'][1]- h_exit)/(h-h_exit)
    delta_phi   = (h - h_exit)*r_exit
    # gamma_heave is the safety factor that was used.
    gamma_heave = QS_dictionary['Input']['Safety']['gamma_he'][1]
    gamma_schem = QS_dictionary['Input']['Scenario ' + str(scenario)]['gb,u'][1]
    kwelscherm  = QS_dictionary['Results']['Scenario ' + str(scenario)]['Heave']['kwelscherm aanwezig']
    d_cover     = QS_dictionary['Input']['Scenario ' + str(scenario)]['d,pip'][1]
    h = h-0.5
    if kwelscherm == 'Ja':
        i_crit = 0.5
    else:
        i_crit = 0.3
    if d_cover>0:
        i = delta_phi/d_cover
        # SF = (i_crit/(gamma_schem*gamma_heave))/i
        SF = (i_crit/(gamma_schem))/i

        if SF >=1:
            assessment = 'voldoende'
        else:
            assessment = 'onvoldoende'
    else:
        SF = 0
        assessment = 'geen deklaag'
    return assessment, SF


def upliftAssessment(QS_dictionary, scenario):
    # function for the assessment of uplift
    h           = QS_dictionary['Input']['Scenario ' + str(scenario)]['j 0 '][1]
    h_exit      = QS_dictionary['Input']['Scenario ' + str(scenario)]['hp = j 3'][1]
    r_exit      = (QS_dictionary['Input']['Scenario ' + str(scenario)]['j2'][1]- h_exit)/(h-h_exit)
    delta_phi   = (h - h_exit)*r_exit
    d_cover     = QS_dictionary['Input']['Scenario ' + str(scenario)]['d,pot'][1]
    gamma_sat   = 16.5 #QS_dictionary['Input']['Scenario ' + str(scenario)]['Gemiddeld volumegewicht:'][1]
    gamma_water = 9.81
    gamma_up    = QS_dictionary['Input']['Safety']['gamma_up'][1]
    gamma_schem = QS_dictionary['Input']['Scenario ' + str(scenario)]['gb,u'][1]
    h = h - 0.5
    if d_cover>0:
        dphi_crit = (d_cover*(gamma_sat-gamma_water))/gamma_water
        #Beware of the safety factors: for the assessment you need to put it in here, if you want to approach the failure probability you shouldn't
        # SF = (dphi_crit/(gamma_up*gamma_schem))/delta_phi
        SF = (dphi_crit/(gamma_schem))/delta_phi
        if SF >=1:
            assessment = 'voldoende'
        else:
            assessment = 'onvoldoende'
    else:
        SF = 0
        assessment = 'geen deklaag'
    return assessment, SF

def pipingAssessment(QS_dictionary, scenario):
    # function for the assessment of piping

    h           = QS_dictionary['Input']['Scenario ' + str(scenario)]['j 0 '][1]
    h_exit      = QS_dictionary['Input']['Scenario ' + str(scenario)]['hp = j 3'][1]
    d_cover     = QS_dictionary['Input']['Scenario ' + str(scenario)]['d,pip'][1]
    L           = QS_dictionary['Input']['Scenario ' + str(scenario)]['L1:pip'][1] + QS_dictionary['Input']['Scenario ' + str(scenario)]['L2'][1] + QS_dictionary['Input']['Scenario ' + str(scenario)]['L3'][1]
    D           = QS_dictionary['Input']['Scenario ' + str(scenario)]['D'][1]
    theta       = 37.0
    d70         = QS_dictionary['Input']['Scenario ' + str(scenario)]['d70'][1]
    k           = QS_dictionary['Input']['Scenario ' + str(scenario)]['kzand,pip'][1]
    gamma_pip   = QS_dictionary['Input']['Safety']['gamma_pip'][1]
    gamma_schem = QS_dictionary['Input']['Scenario ' + str(scenario)]['gb,u'][1]
    m_p = 1
    h = h - 0.5
    Z, dh, dh_c = Mechanisms.LSF_sellmeijer(h, h_exit, d_cover, L, D, theta, d70, k, m_p)
    # SF = (dh_c/(gamma_pip*gamma_schem))/dh
    # SF = (dh_c/(gamma_pip))/dh
    SF = dh_c/dh
    if SF < 1:
        assessment = 'onvoldoende'
    else:
        assessment = 'voldoende'

    return assessment, SF

allsections = {}
sectionnames = []
# # return dict data to new dict
for i in onlyfiles:
    allsections[i.split(' ')[1][:-5]] = ld_readDicts(pad + '\\output\\' + i.split(' ')[1][:-5] + '.dta')
    sectionnames.append(i.split(' ')[1][:-5])
    Tnorm = 10000 #Varies per section, you could also make this an input parameter depending on the dike segment considered
    assessment_results = {}
    heave = {}
    uplift = {}
    piping = {}
    for scenario in range(1,7):
        # Do assessments for all submechanisms for all scenario's
        # For each part there is a piece of commented code that you can use to check if your safety factors are the same as in the Excel
        if 'Scenario ' +str(scenario) in allsections[i.split(' ')[1][:-5]]['General'].keys():

            #Heave
            assessment, SFh = heaveAssessment(allsections[i.split(' ')[1][:-5]],scenario)
            # print(assessment + '   ' + allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Heave']['toets op Heave'])
            # if not isinstance(allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Heave']['optredend heave gradient'],str):
            #     SF_excel = (allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Heave']['toelaatbaar verhang']/ (allsections[i.split(' ')[1][:-5]]['Input']['Safety']['gamma_he'][1]*allsections[i.split(' ')[1][:-5]]['Input']['Scenario ' + str(scenario)]['gb,u'][1]))/allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Heave']['optredend heave gradient']
            # else:
            #     SF_excel = 0
            # print(str(SF) + '   ' + str(SF_excel))
            # print(i)
            heave['Scenario ' + str(scenario)] = {}
            heave['Scenario ' + str(scenario)]['oordeel'] = assessment
            heave['Scenario ' + str(scenario)]['SF'] = SFh
            if SFh > 0:
                heave['Scenario ' + str(scenario)]['beta_cross'] = (1/0.46)*(np.log(SFh/0.48)+0.27*-norm.ppf(1/Tnorm))
            else:
                heave['Scenario ' + str(scenario)]['beta_cross'] = 0


            # #Uplift
            assessment, SFu = upliftAssessment(allsections[i.split(' ')[1][:-5]],scenario)
            uplift['Scenario ' + str(scenario)] = {}
            uplift['Scenario ' + str(scenario)]['oordeel'] = assessment
            uplift['Scenario ' + str(scenario)]['SF'] = SFu
            if SFu>0:
                uplift['Scenario ' + str(scenario)]['beta_cross'] = (1/0.48)*(np.log(SFu/0.37)+0.30*-norm.ppf(1/Tnorm))
            else:
                heave['Scenario ' + str(scenario)]['beta_cross'] = 0

            #Piping
            assessment, SFp = pipingAssessment(allsections[i.split(' ')[1][:-5]],scenario)
            # print(assessment + '   ' + allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Piping']['oordeel'])
            # if not isinstance(allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' + str(scenario)]['Piping']['kritiek verval'], str):
            #     SF_excel = allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Piping']['kritiek verval']/allsections[i.split(' ')[1][:-5]]['Results']['Scenario ' +str(scenario)]['Piping']['optredend verval']
            # else:
            #     SF_excel = str('not calculated')
            # print(str(SF) + '   ' + str(SF_excel))
            # print(i)
            piping['Scenario ' + str(scenario)] = {}
            piping['Scenario ' + str(scenario)]['oordeel'] = assessment
            piping['Scenario ' + str(scenario)]['SF'] = SFp
            piping['Scenario ' + str(scenario)]['beta_cross'] = (1/0.37)*(np.log(SFp/1.04)+0.43*-norm.ppf(1/Tnorm))

            assessment_results['Piping'] = piping
            assessment_results['Uplift'] = uplift
            assessment_results['Heave']  = heave
    allsections[i.split(' ')[1][:-5]]['Assessment'] = assessment_results

plotBetaCS(allsections,pad)
# plotPfCS(allsections,pad)


# writeTable(allsections)

#
headers = ['Section Name','Traject Start', 'Traject End', 'Cross section', 'km start', 'km end', 'SF piping', 'SF heave', 'SF uplift', 'beta_cs']
data = []
data.append(headers)
dijkpaalafstand = 0.200
DVlast = []

for i in range(0,len(sectionnames)):
    sectiondict = allsections[sectionnames[i]]
    beta_cs = max(sectiondict['Assessment']['Piping']['Scenario 1']['beta_cross'],sectiondict['Assessment']['Heave']['Scenario 1']['beta_cross'],sectiondict['Assessment']['Uplift']['Scenario 1']['beta_cross'])
    startkm = float(sectiondict['General']['Traject start'][2:5] + '.' + sectiondict['General']['Traject start'][6:]) * dijkpaalafstand
    endkm = float(sectiondict['General']['Traject end'][2:5] + '.' + sectiondict['General']['Traject end'][6:]) * dijkpaalafstand
    if sectionnames[i][:4] == DVlast:
        cs_last = float(allsections[sectionnames[i-1]]['General']['Cross section'][2:5] + '.' + re.findall('\d+',allsections[sectionnames[i-1]]['General']['Cross section'][6:])[0])
        cs_new = float(sectiondict['General']['Cross section'][2:5] + '.' + re.findall('\d+',sectiondict['General']['Cross section'][6:])[0])
        startkm = np.mean((cs_last,cs_new))*dijkpaalafstand
        j = len(data)
        data[j-1][5] = startkm
    data.append([sectionnames[i],  \
                sectiondict['General']['Traject start'], \
                sectiondict['General']['Traject end'], \
                sectiondict['General']['Cross section'], \
                startkm, endkm, \
                sectiondict['Assessment']['Piping']['Scenario 1']['SF'], \
                sectiondict['Assessment']['Heave']['Scenario 1']['SF'], \
                sectiondict['Assessment']['Uplift']['Scenario 1']['SF'], \
                beta_cs])
    DVlast = sectionnames[i][:4]
with open(pad+ r"\output\assessment.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel',lineterminator='\r',delimiter=';')
    wr.writerows(data)


writeDataforNelle()

# allsections = ld_readDicts('C:/Users/Lee/Desktop/test2.dta')
