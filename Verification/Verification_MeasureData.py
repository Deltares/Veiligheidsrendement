from DikeTraject import DikeTraject
from Solutions import Solutions
import pandas as pd
import numpy as np
from pathlib import Path
from Strategy import GreedyStrategy

#THIS SCRIPT OBTAINS RELIABILITY OF ALL OPTIONS FOR THE BASE DATA
def main():
    # PATH = Path(r'c:\PythonResults\SAFE_results\OptimizationBatch\BaseData')
    # PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\98_Papers\Journal\XXXX_Verification'
    #             r'&ApplicationGreedyOptimizationMethodforplanningFloodDefenceReinforcements_CACAIE\Berekeningen\Batch_general_run1\BaseData')
    PATH = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\03_Cases\01_Rivierenland SAFE\WJKlerk\SAFE\data\OptimizationBatch_Overflow\BaseData')
    T = [0, 19, 20, 50, 75, 100]
    STARTYEAR = 2020
    #First we make 1 big TrajectObject
    #Read the data
    TrajectObject = DikeTraject('')
    TrajectObject.ReadAllTrajectInput(PATH, '', T, STARTYEAR)

    #Step 1: Assess current situation
    TrajectObject.runFullAssessment()
    TrajectObject.setProbabilities()
    assessment_directory = PATH.joinpath('AssessmentFigures','Initial')
    TrajectObject.plotAssessmentResults(directory=assessment_directory)
    assessment_directory = PATH.joinpath('AssessmentFigures')
    TrajectObject.plotAssessment(PATH=assessment_directory)

    #Step 2: Evaluate solutions
    solutions_directory = PATH.joinpath('Solutions')
    SolutionsCollection = {}
    for i in TrajectObject.Sections:
        SolutionsCollection[i.name] = Solutions(i)
        SolutionsCollection[i.name].fillSolutions(PATH.joinpath(i.name + '.xlsx'))
        SolutionsCollection[i.name].evaluateSolutions(i, TrajectObject.GeneralInfo, geometry_plot=False,
                                                      plot_dir=solutions_directory.joinpath(i.name), preserve_slope =
                                                      True)
        SolutionsCollection[i.name].SolutionstoDataFrame(filtering='off',splitparams=True)
    pass
    BaseDataStrategy = GreedyStrategy('TC')
    # Combine available measures
    BaseDataStrategy.combine(TrajectObject, SolutionsCollection, filtering='off', splitparams=True)


    for j in BaseDataStrategy.options:
        BaseDataStrategy.options[j].to_csv(PATH.joinpath('Solutions', j + '_Options_TC.csv'))

if __name__ == '__main__':
    main()