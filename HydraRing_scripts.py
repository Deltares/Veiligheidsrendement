import subprocess
import mmap
import pandas as pd
import matplotlib.pyplot as plt
pad = r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Belasting_HydraRing'

def runHydraRing(inifile):
    exelocation = r'C:\Program Files (x86)\WTI\Ringtoets 17.2.2.13491\bin\HydraRing\MechanismComputation.exe'
    subprocess.run([exelocation, inifile],cwd=inifile[:-5])

# runHydraRing(r'D:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Belasting_HydraRing\1.ini')


def readDesignTable(filename):
    import re
    values = [];
    count = 0
    f = open(filename,'r')
    for line in f:
        if count == 0:
            headers = re.split("  +",line)[1:]
        else:
            val = re.split("  +",line)[1:]
            val = [i.replace('\n', '') for i in val]
            val = [float(i) for i in val]
            values.append(val)


        count+=1
    data = pd.DataFrame(values, columns=headers)
    # data.convert_objects(convert_numeric=True).dtypes
    f.close()
    return data
def plotDesignTable(data):
    data.plot(x='Value', y='Failure probability',kind='line')
    plt.ylabel(r'Exceedance Probability')
    plt.legend(loc='upper right')
    # plt.ylim(ymin=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

data = readDesignTable(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\Belasting_HydraRing\designTable.txt')
plotDesignTable(data)
print(data)