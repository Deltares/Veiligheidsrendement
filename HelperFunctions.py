try:
    import cPickle as pickle
except:
    import pickle
from openturns.viewer import View
import matplotlib.pyplot as plt
import openturns as ot
from pandas import DataFrame
import numpy as np
# write to file with cPickle/pickle (as binary)
def ld_writeObject(filePath,object):
    f=open(filePath,'wb')
    newData = pickle.dumps(object, 1)
    f.write(newData)
    f.close()

#Used to read a pickle file which can objects
def ld_readObject(filePath):
    # This script can load the pickle file so you have a nice object (class or dictionary
    f=open(filePath,'rb')
    data = pickle.load(f)
    f.close()
    return data

#Helper function to flatten a nested dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def drawAlphaBarPlot(resultList,xlabels = None, Pvalues = None, method = 'MCS', suppress_ind = None, title = None):
    #draw stacked bars for the importance factors of a list of openturns FORM and results
    idx = np.nonzero(Pvalues)[0][0]
    if method == 'MCS':
        labels = resultList[idx].getImportanceFactors().getDescription()
    elif method == 'FORM':
        labels = resultList[idx].getImportanceFactors(ot.AnalyticalResult.CLASSICAL).getDescription()

    alphas = []
    for i in range(idx, len(resultList)):
        alpha = list(resultList[i].getImportanceFactors(ot.AnalyticalResult.CLASSICAL)) if method == 'FORM' else list(resultList[i].getImportanceFactors())
        if suppress_ind != None:
            for ix in suppress_ind:
                alpha[ix] = 0.0
        alpha = np.array(alpha)/np.sum(alpha)
        alphas.append(alpha)

    alfas = DataFrame(alphas, columns=labels)
    alfas.plot.bar(stacked=True, label=xlabels)
    # print('done')
    if Pvalues != None:
        plt.plot(range(0, len(xlabels)), Pvalues, 'b', label='Fragility Curve')

    xlabels = ['{:4.2f}'.format(xlabels[i]) for i in range(0, len(xlabels))]
    plt.xticks(range(0, len(xlabels)), xlabels)
    plt.legend()
    plt.title(title)
    plt.show()
    #TO BE DONE: COmpare a reference case to Matlab
