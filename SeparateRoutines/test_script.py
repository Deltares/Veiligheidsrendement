import cProfile
import pstats
import io
from pathlib import Path

path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\01_Scripts\SAFE')
s = io.StringIO()
ps = pstats.Stats(str(path.joinpath('SAFE','InvestmentsSAFE.profile')),stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())