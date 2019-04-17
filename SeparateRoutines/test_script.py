# import shelve
import pandas as pd
import matplotlib.pyplot as plt
# # padnaam = "d:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\03_Cases\\01_Rivierenland SAFE\\Local\\Nelle_Resultaten\\SAFE_16-3_Run5_geenLE\\Case_SAFE"
# padnaam = "d:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\03_Cases\\01_Rivierenland SAFE\\Local\\Nelle_Resultaten\\SAFE_16-4_Run7_geenLE\\Case_SAFE"
#
# # #Store intermediate results:
# filename = padnaam + '\\FINALRESULT.out'
#
# #open shelf
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()
# TestCaseStrategyTC.Probabilities[0].to_csv(padnaam + "\\Betas.csv")


pathname = 'd:\\wouterjanklerk\\My Documents\\00_PhDgeneral\\03_Cases\\01_Rivierenland SAFE\\Local\\HRING basisbestanden\\'
file = 'hoogtecheck16_4.xlsx'
data_input = pd.read_excel(pathname + file,sheet_name='doorsnedes')
data_all   = pd.read_excel(pathname + file,sheet_name='traject')
data_bounds = pd.read_excel(pathname + file,sheet_name='vakgrenzen')
plt.plot(data_all['x'], data_all['hoogte'], label = 'data HKV')
plt.plot(data_input['x'], data_input['hoogte'], label = 'gebruikte doorsnedes',marker='o',linestyle='')

for i in range(0,len(data_input)):
    plt.hlines(data_input['hoogte'][i],data_input['lower'][i],data_input['upper'][i],colors='k',linestyles='dashed')
    plt.hlines(data_input['HBN5l'][i],data_input['lower'][i],data_input['upper'][i],colors='r',linestyles='dashed')
plt.vlines(data_bounds['x'],6 ,8.5,linestyles='dotted')
plt.xlabel('km')
plt.ylabel('hoogte')
plt.legend()
plt.grid()
plt.show()
print()
