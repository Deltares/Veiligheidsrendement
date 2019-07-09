import csv
from pathlib import Path
import pandas as pd
def compare_up_to_investment_limit(wb_TC, wb_OI, investment_limit):
    data = []
    investment_TC = 0
    investment_OI = 0
    i = 0

    while investment_TC or investment_OI <= investment_limit:
        print(i)
        # investment_TC += wb_TC[i][]
        # investment_OI += wb_OI[i][]
        i += 1

def read_csv(path):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []

        for row in csv_reader:
            data.append(row)

    return data

def main():
    ## GENERAL SETTINGS
    traject = '16-4'
    casename = 'SAFE'
    path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\SAFE\SAFE_' + traject + '_geenLE')
    directory = path.joinpath('Case_' + casename, 'results')
    TC_file = 'TakenMeasures_TC.csv'
    OI_file = 'TakenMeasures_OI.csv'
    investment_limit = 15000000

    #Open TC and OI files
    wb_TC = pd.DataFrame.from_csv(directory.joinpath(TC_file))
    wb_OI = pd.DataFrame.from_csv(directory.joinpath(OI_file))

    wb_TC['SummedLCC'] = np.cumsum(wb_TC['LCC'])
    wb_OI['SummedLCC'] = np.cumsum(wb_OI['LCC'])

    #Compare TC and OI upto investment limit
    compare_up_to_investment_limit(wb_TC, wb_OI, investment_limit)

    print('hi')
if __name__ == '__main__':
    main()
