try:
    import cPickle as pickle
except:
    import pickle

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
