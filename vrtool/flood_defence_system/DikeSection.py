import pandas as pd

from vrtool.FloodDefenceSystem.SectionReliability import SectionReliability


# class that contains a DikeProfile consisting of 4 or 6 characteristic points:
class DikeProfile:
    def __init__(self,name = None):
        self.characteristic_points = {}
        self.name = name
    def add_point(self,key, xz):
        self.characteristic_points[key] = xz

    def generate_shapely_polygon(self):
        pass

    def read_points(self):
        pass

    def to_csv(self,path):
        #add mkdir?
        pd.DataFrame.from_dict(self.characteristic_points, orient='index', columns=['x', 'z']).to_csv(path.joinpath('{}.csv'.format(self.name)))

#initialize the DikeSection class, as a general class for a dike section that contains all basic information
class DikeSection:
    def __init__(self, name, traject):
        self.Reliability = SectionReliability()
        self.name = name  #Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
        # Basic traject info TODO: THIS HAS TO BE MOVED TO TRAJECT OBJECT
        self.TrajectInfo = {}
        if traject == '16-4':
            self.TrajectInfo['TrajectLength'] = 19480
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
        elif traject == '16-3':
            self.TrajectInfo['TrajectLength'] = 19899
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
        elif traject == '38-1':
            self.TrajectInfo['TrajectLength'] = 28902
            self.TrajectInfo['Pmax'] = 1. / 10000; self.TrajectInfo['omegaPiping'] = 0.24; self.TrajectInfo['aPiping'] = 0.9; self.TrajectInfo['bPiping'] = 300
    def readGeneralInfo(self, path, sheet_name):
        #Read general data from sheet in standardized xlsx file
        df = pd.read_excel(path.joinpath(self.name + ".xlsx"), sheet_name=None)

        for name, sheet_data in df.items():
            if name == sheet_name:
                data = df[name].set_index(list(df[name])[0])
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
        setattr(self, 'InitialGeometry', df['Geometry'].set_index(list(df['Geometry'])[0]))