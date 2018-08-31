## This script is made based on the Quick Scan Excel files from traject 16-3 and 16-4.
## It reads all excelsheets in a specified folder and writes them as output to a set of 'Pickle' files.
## Author: Wouter Jan Klerk
## Date: 20180501


#Import some needed libraries
from os import listdir
from os.path import isfile, join
from openpyxl import load_workbook
import itertools
try:
    import cPickle as pickle
except:
    import pickle

#Define the path where the Excelsheets are located
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\16-3'


# write to file with cPickle/pickle (as binary)
def ld_writeDicts(filePath,dict):
    f=open(filePath,'wb')
    newData = pickle.dumps(dict, 1)
    f.write(newData)
    f.close()

# read file decoding with cPickle/pickle (as binary)
def ld_readDicts(filePath):
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data
# # return dict data to new dict
# newDataDict = ld_readDicts('C:/Users/Lee/Desktop/test2.dta')


#This function replaces all the major greek symbols with normal ascii characters, as otherwise accessing the dictionary keys becomes a hassle
def print_dict(d):
    new = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = print_dict(v)
        # if k.find(chr(947)) is not -1:
        k = k.replace(chr(945), 'alpha_')  if k.find(chr(945)) is not -1 else k #alfa
        k = k.replace(chr(946), 'beta_')   if k.find(chr(946)) is not -1 else k #beta
        k = k.replace(chr(947), 'gamma_')  if k.find(chr(947)) is not -1 else k #gamma
        k = k.replace(chr(948), 'delta_')  if k.find(chr(948)) is not -1 else k #delta
        k = k.replace(chr(955), 'lambda_') if k.find(chr(955)) is not -1 else k #lambda
        k = k.replace(chr(956), 'mu_')     if k.find(chr(956)) is not -1 else k #mu
        k = k.replace(chr(966), 'phi_')    if k.find(chr(966)) is not -1 else k #phi
        k = k.replace(chr(969), 'omega_')  if k.find(chr(969)) is not -1 else k #omega
        new[k]=v
    return new

#Define a range of characters
def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

#This script reads input and results specific for a scenario
def readScenario(scenario, wb):
    #Read all the input
    input = {};     results = {}
    sheet = wb["Scenario_" + str(scenario)]
    for i in itertools.chain(range(10,42),range(66,67)):
        if sheet['C' + str(i)].value is not None:
            if sheet['B' + str(i)].value is not None:
                input[sheet['B'+str(i)].value] = (sheet['A' + str(i)].value,sheet['C'+str(i)].value)
            elif sheet['A' + str(i)].value is not None:
                input[sheet['A'+str(i)].value] = (sheet['A' + str(i)].value,sheet['C'+str(i)].value)
    for i in [20, 21,24,27]:
        if sheet['J' + str(i)].value is not None:
            if sheet['I' + str(i)].value is not None:
                input[sheet['I'+str(i)].value] = (sheet['H' + str(i)].value,sheet['J'+str(i)].value)
            elif sheet['I' + str(i)].value is not None:
                input[sheet['I'+str(i)].value] = (sheet['I' + str(i)].value,sheet['J'+str(i)].value)

    # Read the main results
    piping = {}; uplift = {}; heave = {};
    #Uplift
    for i in [28,29,30]:
        if sheet['J' + str(i)].value is not None:
            if sheet['I' + str(i)].value is not None:
                uplift[sheet['I'+str(i)].value] = (sheet['H' + str(i)].value,sheet['J'+str(i)].value)
            elif sheet['H' + str(i)].value is not None:
                uplift[sheet['H'+str(i)].value] = (sheet['H' + str(i)].value,sheet['J'+str(i)].value)

    #Heave
    for i in range(70,77):
        heave[sheet['I' + str(i)].value] = sheet['J' + str(i)].value

    #Piping
    sheet = wb["Rekenbladinvoer"]
    for i in itertools.chain(char_range('O', 'U')):
        piping[sheet['B' + i+str(42)].value] = sheet['B' +i+str(44+scenario)].value
    results['Heave'] = heave
    results['Uplift'] = uplift
    results['Piping'] = piping

    return input, results

# This is the main script for reading. It directs to all the scenarios (usually only 1 is present) and writes general info on the dike section.
def readQSExcel(pathname,file):
    wb = load_workbook(filename = pathname + "\\" + file, data_only=True)
    data = {}
    sectioninfo = {}
    sheet = wb["Geometrie_DWP"]
    #Read general info on the dike section
    sectioninfo['Traject start'] = sheet['B3'].value
    sectioninfo['Traject end']   = sheet['D3'].value
    sectioninfo['Cross section'] = sheet['B4'].value
    data['General'] = sectioninfo

    safety = {}
    sheet = wb["Sterktefactor"]
    for i in [11, 12, 13, 14, 19, 20, 21, 22, 23, 27]:
        safety[sheet['B' + str(i)].value] = (sheet['A' + str(i)].value, sheet['C' + str(i)].value)
    input = {}; results = {}
    input['Safety'] = safety

    #Determine number of active scenarios
    for i in range (0,6):
        sheet = wb["Scenario_kansen_Pf"]
        line = 'B' + str(27+i)
        if sheet[line].value > 0.:
            scen_num = sheet['B'+str(27+i)].value
            sectioninfo['Scenario ' + str(scen_num)] = sheet['I' +str(27+i)].value
            input['Scenario ' + str(scen_num)], results['Scenario ' + str(scen_num)] = readScenario(scen_num, wb)
            results['Scenario ' + str(scen_num)]['PS_s,i'] = sheet['B'+str(27+i)].value
            results['Scenario ' + str(scen_num)]['gamma_s,i'] = sheet['C'+str(27+i)].value
            results['Scenario ' + str(scen_num)]['Pf_s,i'] = sheet['E' + str(27 + i)].value

    data['Input'] = input
    data['Results'] = results
    data_out = print_dict(data)

    return data_out







#Make a list of all the files (careful: there should only be Excelsheets in the path, and if you open them you might get
#  an error (there appears a file called ~section.xlsx which the script doesnt understand
onlyfiles = [f for f in listdir(pad) if isfile(join(pad,f))]
for i in range(0,len(onlyfiles)):
    data = readQSExcel(pad, onlyfiles[i])
    ld_writeDicts(pad + '\\output\\' + onlyfiles[i].split(' ')[1][:-5] + '.dta', data)
    print(onlyfiles[i].split(' ')[1][:-5])
    data = {}


#if you want to read them back in you can use this line
# testdata = ld_readDicts(pad + '\\output\\' + onlyfiles[i].split(' ')[1][:-5] + '.dta')
