import shutil
import os
import zipfile
from pathlib import Path


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
            
PATH = []
resultpath = Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/')
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Normal_cautious_f=1.5'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Overflow_cautious_f=1.5'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/AllCases_cautious_f=1.5'))

PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Overflow_cautious_f=3'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Normal_cautious_f=3'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/AllCases_cautious_f=3'))

PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Overflow_robust'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Normal_robust'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/AllCases_robust'))

PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Overflow_combined'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/Batch_Normal_combined'))
PATH.append(Path(r'd:/wouterjanklerk/My Documents/00_PhDgeneral/98_Papers/Journal/XXXX_SAFEGreedyMethod_CACAIE/Berekeningen/AllCases_cautious_combined'))

PATH.append(Path(r'C:/Users/wouterjanklerk/NaarHorizon/TestComputationTime'))


if __name__ == '__main__':
    # for path in PATH:
    #     for i in path.iterdir():
    #         if i.is_dir():
    #             caseset = i.name
    #             shutil.make_archive(Path(i.parent.parent).joinpath('zip-files',i.parent.name,caseset), 'zip', i)
    for path in PATH:
        for i in path.iterdir():
            if not i.is_dir():
                filename = i.name
                destination  =Path(i.parent.parent).joinpath('resultaten',i.parent.name)
                destination.mkdir(parents=True,exist_ok=True)
                shutil.copy(i, destination.joinpath(filename))




