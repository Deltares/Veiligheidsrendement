import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from ProbabilisticFunctions import TableDist

#script to run HydraRing. exelocation should refer to the location of MechanismComputation.exe.
def runHydraRing(inifile, exelocation = r'C:\Program Files (x86)\BOI\Riskeer 21.1.1.2\Application\Standalone\Deltares\HydraRing-20.1.3.10236\MechanismComputation.exe'):
    subprocess.run([exelocation, str(inifile)],cwd=str(inifile.parent))

#write a design table to a OpenTurns TableDist as defined by ProbabilisticFunctions.TableDist
def DesignTableOpenTurns(filename, gridpoints=2000):
    data = readDesignTable(filename)
    wls = list(data.iloc[:, 0])
    #print('Warning water levels reduced')
    #wls = np.subtract(wls, 0.5)
    p_nexc = list(1-data.iloc[:, 1])
    h = TableDist(np.array(wls), np.array(p_nexc), extrap=True, isload=True)
    h = ot.Distribution(TableDist(np.array(wls), np.array(p_nexc), extrap=True, isload=True, gridpoints=gridpoints))
    return h

def readDesignTable(filename):
    import re
    values = []
    count = 0
    f = open(filename, 'r')
    for line in f:
        if count == 0:
            headers = re.split("  +", line)[1:]
        else:
            val = re.split("  +", line)[1:]
            val = [i.replace('\n', '') for i in val]
            val = [float(i) for i in val]
            values.append(val)
        count += 1
    data = pd.DataFrame(values, columns=headers)
    f.close()
    return data

#script to plot a value-exceedance probability graph based on a designtable from HydraRing.
def plotDesignTable(data):
    data.plot(x='Value', y='Failure probability',kind='line')
    plt.ylabel(r'Exceedance Probability')
    plt.legend(loc='upper right')
    # plt.ylim(ymin=0)
    plt.yscale('log')
    plt.tight_layout()

def main():
    pass

if __name__ == "__main__":
    main()