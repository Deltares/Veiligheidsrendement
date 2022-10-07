import os
import shutil
from pathlib import Path

PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen')
def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.rpartition('.')[0]
        format = base.rpartition('.')[2]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)

for i in PATH.iterdir():
    if 'Batch' in i.name:
        make_archive(str(i), str(i.parent.joinpath(i.name + '.zip')))
