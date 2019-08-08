#This script makes the entire input structure from the general input files
import pandas as pd
from HydraRing_scripts import readDesignTable
from openpyxl import load_workbook
from pathlib import Path
from shutil import copyfile

def main():
    traject = 'Test'
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland '
                r'SAFE\WJKlerk\SAFE\data\InputFiles\OptimizationTestCase')
    filename = 'Dijkvakindeling_v4.2_Optimalisatie.xlsx'
    path_WLRise_HBNRise = False
    #Comment this out if it has to be read from the
    # path_WLRise_HBNRise = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland '
    #             r'SAFE\WJKlerk\SAFE\data\InputFiles\SAFEInput\HBN HKV')
    # filename_WLRise_HBNRise = 'Resultaten_OI2014v4_BeleidsmatigeAfvoerverdeling_20180430.xlsx'

    #Open en read data
    df = pd.read_excel(path.joinpath(filename), sheet_name=None)

    if path_WLRise_HBNRise:
        df_WL = pd.read_excel(path_WLRise_HBNRise.joinpath(filename_WLRise_HBNRise), sheet_name='Overzicht_Werkelijk', header=1, usecols='A, G:L')
        df_WL['Dijkpaal'] = df_WL['Dijkpaal'].str.replace('.', '')
        df_HBN = pd.read_excel(path_WLRise_HBNRise.joinpath(filename_WLRise_HBNRise), sheet_name='Overzicht_Werkelijk', header=1, usecols='A, AF:AK')
        df_HBN['Dijkpaal'] = df_HBN['Dijkpaal'].str.replace('.', '')

    DikeSections = df['Dijkvakindeling_keuze_info'].rename(columns=df['Dijkvakindeling_keuze_info'].iloc[0]).drop(df['Dijkvakindeling_keuze_info'].index[0])
    DikeSections = DikeSections[((DikeSections['Traject'] == traject) & (DikeSections['Wel of niet meerekenen'] == 1))].reset_index(drop=True)
    STBI_data = df['Info voor STBI'].iloc[:, [1, 6, 7, 8, 9]]
    Piping_data = df['Info voor Piping'].iloc[:, [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    Housing = df['Info voor huizen'].iloc[:, [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    measures = pd.read_csv(path.joinpath(traject, 'Input/measures.csv'), delimiter=';')

    General = {}
    General['Name'] = ['Length', 'Start', 'End', 'Overflow', 'StabilityInner', 'Piping', 'LoadData', 'YearlyWLRise', 'HBNRise_factor']
    General['Type'] = ['', '', '', 'Simple', 'Simple', 'SemiProb', '', '', '']

    #Make folders if not exist:
    if not path.joinpath(traject, 'Output').is_dir():
        path.joinpath(traject, 'Output/StabilityInner').mkdir(parents=True, exist_ok=True)
        path.joinpath(traject, 'Output/Piping').mkdir(parents=True, exist_ok=True)
        path.joinpath(traject, 'Output/Overflow').mkdir(parents=True, exist_ok=True)
        path.joinpath(traject, 'Output/Toetspeil').mkdir(parents=True, exist_ok=True)

    if path_WLRise_HBNRise:
        for i in DikeSections.index:
            df_YearlyWLRise = df_WL[df_WL['Dijkpaal'] == DikeSections['Dwarsprofiel HoogteToetspeil'][i]]
            YearlyWLRise_factor = float((df_YearlyWLRise[2125].values - df_YearlyWLRise[2015].values) / (2125 - 2015))
            df_HBNRise_factor = df_HBN[df_HBN['Dijkpaal'] == DikeSections['Dwarsprofiel HoogteToetspeil'][i]]
            HBNRise_factor = float((df_HBNRise_factor[2125].values - df_HBNRise_factor[2015].values) / (2125 - 2015)) / YearlyWLRise_factor

            #Write YearlyWLRise and HBNRise factors to mastersheet
            wb = load_workbook(path.joinpath(filename))
            ws = wb.get_sheet_by_name('Dijkvakindeling_keuze_info')

            for row in range(ws.max_row):
                if ws[row + 1][9].value == DikeSections['Dwarsprofiel HoogteToetspeil'][i]:
                    ws.cell(row=row + 1, column=14).value = YearlyWLRise_factor
                    ws.cell(row=row + 1, column=15).value = HBNRise_factor
                else:
                    pass

            wb.save(path.joinpath(filename))

    for i in DikeSections.index:
        HBN_basis = pd.read_csv(path.joinpath(traject, 'Input/base_HBN.csv'), delimiter=';')

        General['Value'] = [DikeSections['Lengte dijkvak'][i], DikeSections['Van'][i], DikeSections['Tot'][i],
                            'Overflow/' + DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '_Overflow.csv',
                            'StabilityInner/' + DikeSections['Dwarsprofiel STBI/STBU'][i] + '_StabilityInner.csv',
                            'Piping/' + DikeSections['Dwarsprofiel piping'][i] + '_Piping.csv', 'DESIGNTABLE_' +
                            DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '.txt',
                            DikeSections['Waterstandstijging'][i],
                            DikeSections['HBN factor'][i]]
        profile = pd.read_csv(path.joinpath(traject, 'Input/profiles', DikeSections['Dwarsprofiel Geometrie'][i] + '.csv'), index_col=0)
        toExcel = pd.DataFrame.from_dict(General)[['Name', 'Value', 'Type']]
        houses_data_location = Housing[((Housing['Naam dijkvak'] == DikeSections['dv_nummer'][i]) & (Housing['Naam traject'] == traject))].transpose().drop(['Naam traject', 'Naam dijkvak'], axis=0).reset_index()
        houses_data_location.columns = ['distancefromtoe', 'number']

        #Write Data
        writer = pd.ExcelWriter(path.joinpath(traject, 'Output/DV' + '{:02d}'.format(DikeSections['dv_nummer'][i]) + '.xlsx'))
        toExcel.to_excel(writer, sheet_name='General', index=False)
        measures.to_excel(writer, sheet_name='Measures', index=False)
        profile.to_excel(writer, sheet_name='Geometry', index=False)
        houses_data_location.to_excel(writer, sheet_name='Housing', index=False)
        writer.save()

        #Write stability inner
        STBI_data_location = STBI_data[STBI_data['dwarsprofiel'] == DikeSections['Dwarsprofiel STBI/STBU'][i]].transpose().drop(['dwarsprofiel'], axis=0).reset_index()
        STBI_data_location.columns = ['Name', "Value"]
        STBI_data_location = STBI_data_location.set_index('Name')
        STBI_data_location.to_csv(path.joinpath(traject, 'Output/StabilityInner', DikeSections['Dwarsprofiel STBI/STBU'][i] + '_StabilityInner.csv'))

        #Write piping
        Piping_data_location = Piping_data[Piping_data['dwarsprofiel'] == DikeSections['Dwarsprofiel piping'][i]].transpose().drop(['dwarsprofiel'], axis=0).reset_index()
        Piping_data_location.columns = ['Name', "Value"]
        Piping_data_location = Piping_data_location.set_index('Name')
        Piping_data_location.to_csv(path.joinpath(traject, 'Output/Piping', DikeSections['Dwarsprofiel piping'][i] + '_Piping.csv'))

        #Write overflow
        OverflowData = readDesignTable(path.joinpath(traject, 'Input/designtables_HBN/DESIGNTABLE_' + DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '.txt'))

        if len(OverflowData) != 13:
            HBN_basis = HBN_basis.iloc[0:len(OverflowData)]
            print('Warning! length is not 13!')

        #Overwrite crestheight, betas, h_c and if possible dhc(t)
        # HBN_basis['h_crest'].ix[0] = crestlevels.loc[crestlevels['dijkpaal'] == DikeSections.iloc[i]['Dwarsprofiel HoogteToetspeil']]['hcrest'].values[0]
        HBN_basis['h_crest'][0] = DikeSections['Kruinhoogte'][i]
        HBN_basis['h_c'] = OverflowData['Value'].values
        HBN_basis['beta'] = OverflowData['Beta\n'].values

        if len(DikeSections['Kruindaling'].value_counts()) > 0:
            HBN_basis['dhc(t)'][0] = DikeSections['Kruindaling'][i]

        HBN_basis.to_csv(path.joinpath(traject, 'Output/Overflow', DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '_Overflow.csv'), index=False)

        #Copy designtables
        copyfile(path.joinpath(traject, 'Input/designtables_TP/DESIGNTABLE_' + DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '.txt'), path.joinpath(traject, 'Output/Toetspeil/DESIGNTABLE_' + DikeSections['Dwarsprofiel HoogteToetspeil'][i] + '.txt'))

if __name__ == '__main__':
    main()
