from pathlib import Path

import numpy as np
import pandas as pd


def compare_investment_strategy(wb_TC, wb_OI, investment_limit, output_file_path, measure_file_path):
    investment_TC = 0
    wb_TC['dcrest'] = np.where(wb_TC['dcrest'] == -999, '-', wb_TC['dcrest'])
    wb_TC['dberm'] = np.where(wb_TC['dberm'] == -999, '-', wb_TC['dberm'])
    wb_OI['dcrest'] = np.where(wb_OI['dcrest'] == -999, '-', wb_OI['dcrest'])
    wb_OI['dberm'] = np.where(wb_OI['dberm'] == -999, '-', wb_OI['dberm'])

    # Read measure data
    measures = pd.read_csv(measure_file_path, delimiter=';')

    # Open a new .xlsx file in the result directory
    with pd.ExcelWriter(output_file_path) as writer:
        df_TC = pd.DataFrame(data=wb_OI['Section'].sort_values().tolist(), columns=['Dijkvak'])
        df_TC_below_limit = pd.DataFrame(columns=['Dijkvak', 'Prioritering', 'TC Maatregel Tot Limiet', 'Kosten', 'D_Crest', 'D_Berm'])
        df_TC_above_limit = pd.DataFrame(columns=['Dijkvak', 'TC Maatregel Na Limiet', 'Kosten', 'D_Crest', 'D_Berm'])
        df_TC_below_limit['Dijkvak'] = df_TC['Dijkvak']
        df_TC_above_limit['Dijkvak'] = df_TC['Dijkvak']

        for i, j in enumerate(range(len(wb_TC['Section']))):
            measure_id = wb_TC['ID'][j+1].split('+')

            # Replace the measure ID with the measure description
            if len(measure_id) == 1:
                measure_description = measures[measures['ID'] == int(measure_id[0])]['Name'].values[0]
            else:
                measure_description = measures[measures['ID'] == int(measure_id[0])]['Name'].values[0] + ' + ' + measures[measures['ID'] == int(measure_id[1])]['Name'].values[0]

            # Count investment costs
            investment_TC += wb_TC['LCC'][j+1]

            # Determine limit index
            if investment_TC <= investment_limit:
                index = df_TC_below_limit[df_TC_below_limit['Dijkvak'] == wb_TC['Section'][j+1]].index
                df_TC_below_limit.iloc[index, [2, 4, 5]] = [measure_description, wb_TC['dcrest'][j+1], wb_TC['dberm'][j+1]]
                if df_TC_below_limit.iloc[index, 1].isnull().any().any():
                    df_TC_below_limit.iloc[index, 1] = i+1
                if df_TC_below_limit.iloc[index, 3].isnull().any().any():
                    costs = 0
                else:
                    costs = df_TC_below_limit.iloc[index, 3].values
                costs += wb_TC['LCC'][j+1]
                df_TC_below_limit.iloc[index, 3] = costs
            else:
                index = df_TC_above_limit[df_TC_above_limit['Dijkvak'] == wb_TC['Section'][i+1]].index
                df_TC_above_limit.iloc[index, [1, 3, 4]] = [measure_description, wb_TC['dcrest'][j+1], wb_TC['dberm'][j+1]]
                if df_TC_above_limit.iloc[index, 2].isnull().any().any():
                    costs = 0
                else:
                    costs = df_TC_above_limit.iloc[index, 2].values
                costs += wb_TC['LCC'][j+1]
                df_TC_above_limit.iloc[index, 2] = costs

        # Merge dataframes
        df_TC_below_limit.drop('Dijkvak', axis=1, inplace=True)
        df_TC_above_limit.drop('Dijkvak', axis=1, inplace=True)
        df_TC = pd.concat([df_TC, df_TC_below_limit, df_TC_above_limit], axis=1, ignore_index=True)
        df_TC.fillna('', inplace=True)

        df_OI = pd.DataFrame({'Dijkvak': wb_OI['Section'].sort_values().tolist(), 'OI Maatregel': '', 'Kosten': '', 'D_Crest': '', 'D_Berm': ''})

        for i in range(len(wb_OI['Section'])):
            measure_id = wb_OI['ID'][i+1].split('+')

            # Replace the measure ID with the measure description
            if len(measure_id) == 1:
                measure_description = measures[measures['ID'] == int(measure_id[0])]['Name'].values[0]
            else:
                measure_description = measures[measures['ID'] == int(measure_id[0])]['Name'].values[0] + ' + ' + measures[measures['ID'] == int(measure_id[1])]['Name'].values[0]

            index = df_OI[df_OI['Dijkvak'] == wb_OI['Section'][i+1]].index
            df_OI.iloc[index, 1:] = [measure_description, wb_OI['LCC'][i+1], wb_TC['dcrest'][i+1], wb_TC['dberm'][i+1]]

        df_OI.fillna('', inplace=True)

        # Write data to .xlsx file
        df_TC.to_excel(writer, sheet_name='TC and OI Comparison', header=False, index=False, startrow=1, engine="openpyxl")
        df_OI.to_excel(writer, sheet_name='TC and OI Comparison', header=False, index=False, startrow=1, startcol=11, engine="openpyxl")

        # Add Header Format
        wb = writer.book
        column_header = ['Dijkvak', 'Prioritering', 'TC Maatregel Tot Limiet', 'Kosten', 'D_Crest', 'D_Berm', 'TC Maatregel Na Limiet', 'Kosten', 'D_Crest', 'D_Berm', '', 'Dijkvak', 'OI Maatregel', 'Kosten', 'D_Crest', 'D_Berm']
        header_format = wb.add_format({'bold': True})

        for col_num, value in enumerate(column_header):
            writer.sheets['TC and OI Comparison'].write(0, col_num, value, header_format)

        writer.save()

def main():
    # General settings
    investment_limit = 20000000
    traject = '16-4'
    casename = 'backup'
    path = Path('D:\SAFE\data\SAFE\SAFE_' + traject + '_geenLE_Vakindeling_v5.2')
    directory = path.joinpath('Case_' + casename, 'results')
    measure_path = Path('D:\SAFE\data\InputFiles\SAFEInput').joinpath(traject, 'Input')
    TC_file = 'TakenMeasures_TC.csv'
    OI_file = 'TakenMeasures_OI.csv'
    measure_file = 'measures.csv'
    output_file = 'TC_OI_comparison_traject_' + traject + '_investment_limit_' + str(investment_limit) + '.xlsx'

    # Open TC and OI files
    wb_TC = pd.read_csv(directory.joinpath(TC_file), index_col=[0], skiprows=[1])
    wb_OI = pd.read_csv(directory.joinpath(OI_file), index_col=[0], skiprows=[1])

    # Compare TC and OI investment limit
    compare_investment_strategy(wb_TC, wb_OI, investment_limit, directory.joinpath(output_file), measure_path.joinpath(measure_file))

if __name__ == '__main__':
    main()