from pathlib import Path
from Verification.Verification_SampledCases import BatchRunOptimization
import time


def main():
    PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_SAFEGreedyMethod_CACAIE\Berekeningen\reruns\Overflow_robust')
    # PATH = Path(r'c:\Users\wouterjanklerk\NaarHorizon\Batch_Overflow_cautious_f=1.5\26\047')

    case = '27_042'
    start = time.time()
    # BatchRunOptimization(PATH.joinpath(case),plot_on = False,pareto_on=False,run_MIP=False,GreedySettings={'setting':'cautious','f':1.5,'BCstop':0.1})
    BatchRunOptimization(PATH.joinpath(case),plot_on = True,pareto_on=False,run_MIP=False,GreedySettings={'setting':'robust','f':1.0,'BCstop':0.1})
    # BatchRunOptimization(PATH,plot_on = False,pareto_on=False,run_MIP=False,GreedySettings={'setting':'cautious','f':1.5,'BCstop':0.1})
    print('Runtime: ' + str(time.time()-start) + ' seconds')
    pass




if __name__ == '__main__':
    main()