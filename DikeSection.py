import numpy as np
import pandas as pd
from SectionReliability import SectionReliability

#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.Reliability = SectionReliability()
        self.name = name  #Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
        # Basic traject info NOTE: THIS HAS TO BE REMOVED TO TRAJECT OBJECT
        self.TrajectInfo = {}
        if traject == '16-4':
            self.TrajectInfo['TrajectLength'] = 19480
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
        elif traject == '16-3':
            self.TrajectInfo['TrajectLength'] = 19899
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300

    def readGeneralInfo(self, path, sheet_name):
        #Read general data from sheet in standardized xlsx file
        df = pd.read_excel(path.joinpath(self.name + ".xlsx"), sheet_name=None)

        for name, sheet_data in df.items():
            if name == sheet_name:
                data = df[name].set_index('Name')
                self.MechanismData = {}

                for i in range(len(data)):
                    if data.index[i] == 'Overflow' or data.index[i] == 'Piping' or data.index[i] == 'StabilityInner':
                        self.MechanismData[data.index[i]] = (data.loc[data.index[i]][0], data.loc[data.index[i]][1])
                        # setattr(self, data.index[i], (data.loc[data.index[i]][0], data.loc[data.index[i]][1]))
                    else:
                        setattr(self, data.index[i], (data.loc[data.index[i]][0]))
                        # if data.index[i] == 'YearlyWLRise':
                        #     self.YearlyWLRise = self.YearlyWLRise * 3
                        #     print('Warning: WLRise multiplied!')

            elif name == "Housing":
                self.houses = df['Housing'].set_index('distancefromtoe').rename(columns={'number':'cumulative'})
                # self.houses = pd.concat([df["Housing"], pd.DataFrame(np.cumsum(df["Housing"]['number'].values), columns=['cumulative'])], axis=1, join='inner').set_index(
                #     'distancefromtoe')
            else:
                self.houses = None

        #and we add the geometry
        setattr(self, 'InitialGeometry', df['Geometry'])