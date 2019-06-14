#This script makes the entire input structure from the general input files

import pandas as pd
from shutil import copyfile
from HydraRing_scripts import readDesignTable
import os
traject = '16-4'

# pathname = "d:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\98_Papers\\Conference\\GeoRisk_2019\\Calculations\\GenerateInput\\"
# pathname = "d:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\03_Cases\\01_Rivierenland SAFE\\Local\\All_Input\\" + traject
pathname = "d:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\03_Cases\\01_Rivierenland SAFE\\Local\\Input\\OptimizationInput\\Basic\\"
filename = 'Dijkvakindeling_v4.2_Basic'
DikeSections = pd.read_excel(pathname + '\\Input\\' + filename + '.xlsx',sheet_name='Dijkvakindeling_keuze_info',skiprows=[0])
STBI_data = pd.read_excel(pathname    + '\\Input\\' + filename + '.xlsx',sheet_name='Info voor STBI',usecols="B,G:J")
Piping_data = pd.read_excel(pathname  + '\\Input\\' + filename + '.xlsx',sheet_name='Info voor Piping',usecols="B,G:Q")
Housing = pd.read_excel(pathname      + '\\Input\\' + filename + '.xlsx',sheet_name='Info voor huizen',usecols="A,B,F:O")
measures = pd.read_csv(pathname       + '\\Input\\measures.csv',delimiter=';')
crestlevels = pd.read_excel(pathname  + '\\Input\\InputLocationsHBN.xlsx',sheet_name=traject,usecols="C,G")

DikeSections = DikeSections.loc[DikeSections['Traject']==traject]
DikeSections = DikeSections.loc[DikeSections['Wel of niet meerekenen']==1]
DikeSections = DikeSections.reset_index(drop=True)
General = {}
General['Name']=['Length','Start','End','Overflow','StabilityInner','Piping','LoadData','YearlyWLRise','HBNRise_factor']
General['Type'] =['','','','Simple','Simple','SemiProb','','','']
Profile = {}

#make folders:
if not os.path.exists(pathname + '\\Output\\StabilityInner'):
    os.mkdir(pathname + '\\Output\\StabilityInner')
    os.mkdir(pathname + '\\Output\\Piping')
    os.mkdir(pathname + '\\Output\\Overflow')
    os.mkdir(pathname + '\\Output\\Toetspeil')

for i in DikeSections.index:
    HBN_basis = pd.read_csv(pathname + '\\Input\\base_HBN.csv', delimiter=';')

    General['Value']= [DikeSections.iloc[i]['Lengte dijkvak'],
                       DikeSections.iloc[i]['Van'],
                       DikeSections.iloc[i]['Tot'],
                       '\\Overflow\\' + DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] + '_Overflow.csv',
                       '\\StabilityInner\\' + DikeSections.iloc[i]['Dwarsprofiel STBI/STBU'] + '_StabilityInner.csv',
                       '\\Piping\\' + DikeSections.iloc[i]['Dwarsprofiel piping'] + '_Piping.csv',
                       '\\DESIGNTABLE_' + DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] + '.txt',
                       0.005, 1.5]
    profile = pd.read_csv(pathname + '\\Input\\profiles\\' + DikeSections.iloc[i]['Dwarsprofiel Geometrie'] + '.csv')
    profile = profile.set_index('Unnamed: 0')
    toExcel = pd.DataFrame.from_dict(General)[['Name','Value','Type']]
    opensheet = pd.ExcelWriter(pathname + '\\Output\\DV' + '{:02d}'.format(DikeSections.iloc[i]['dv_nummer']) + '.xlsx')
    toExcel.to_excel(opensheet, sheet_name='General',index=False)
    measures.to_excel(opensheet,sheet_name='Measures',index=False)
    profile.to_excel(opensheet, sheet_name='Geometry',index=False)

    #housin
    houses_data_location = Housing.loc[Housing['Naam dijkvak'] == DikeSections.iloc[i]['dv_nummer']].loc[Housing['Naam traject'] == traject].transpose()
    houses_data_location = houses_data_location.drop(['Naam traject','Naam dijkvak'],axis=0).reset_index()
    houses_data_location.columns = ['distancefromtoe','number']
    houses_data_location.to_excel(opensheet,sheet_name='Housing',index=False)
    opensheet.save()

    #write stability inner
    STBI_data_location = STBI_data.loc[STBI_data['dwarsprofiel']==DikeSections.iloc[i]['Dwarsprofiel STBI/STBU']].transpose()
    STBI_data_location = STBI_data_location.drop(['dwarsprofiel'], axis=0).reset_index()
    STBI_data_location.columns = ['Name', "Value"]
    STBI_data_location = STBI_data_location.set_index('Name')
    STBI_data_location.to_csv(pathname + '\\Output\\StabilityInner\\' + DikeSections.iloc[i]['Dwarsprofiel STBI/STBU'] + '_StabilityInner.csv')

    #write piping
    Piping_data_location = Piping_data.loc[Piping_data['dwarsprofiel']==DikeSections.iloc[i]['Dwarsprofiel piping']].transpose()
    Piping_data_location = Piping_data_location.drop(['dwarsprofiel'], axis=0).reset_index()
    Piping_data_location.columns = ['Name', "Value"]
    Piping_data_location = Piping_data_location.set_index('Name')
    Piping_data_location.to_csv(pathname + '\\Output\\Piping\\' + DikeSections.iloc[i]['Dwarsprofiel piping'] + '_Piping.csv')

    #write overflow
    OverflowData = readDesignTable(pathname + '\\Input\\designtables_HBN\\DESIGNTABLE_'+ DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] +'.txt')
    if len(OverflowData) != 13:
        HBN_basis = HBN_basis.iloc[0:len(OverflowData)]
        print('Warning! length is not 13!')
    #overwrite crestheight, betas and h_c
    # HBN_basis['h_crest'].ix[0] = crestlevels.loc[crestlevels['dijkpaal'] == DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil']]['hcrest'].values[0]
    HBN_basis['h_crest'].ix[0] = DikeSections.Kruinhoogte[i]
    HBN_basis['h_c'] = OverflowData['Value'].values
    HBN_basis['beta'] = OverflowData['Beta\n'].values
    HBN_basis.to_csv(pathname + '\\Output\\Overflow\\' + DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] + '_Overflow.csv',index=False)

    #copy designtables
    copyfile(pathname + '\\Input\\designtables_TP\\DESIGNTABLE_' + DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] + '.txt',
             pathname + '\\Output\\Toetspeil\\DESIGNTABLE_' + DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil'] + '.txt')




#Generate a ton of Excel files

