from pathlib import Path

import pandas as pd

from vrtool.flood_defence_system.section_reliability import SectionReliability


class DikeSection:
    """
    Initialize the DikeSection class, as a general class for a dike section that contains all basic information
    """

    mechanism_data: dict

    def __init__(self, name, traject_name: str):
        self.section_reliability = SectionReliability()
        self.name = name  # Make sure names have the same length by adding a zero. This is non-generic, specific for SAFE
        # Basic traject info TODO: THIS HAS TO BE MOVED TO TRAJECT OBJECT
        self.TrajectInfo = {}
        if traject_name == "16-4":
            self.TrajectInfo["TrajectLength"] = 19480
            self.TrajectInfo["Pmax"] = 1.0 / 10000
            self.TrajectInfo["omegaPiping"] = 0.24
            self.TrajectInfo["aPiping"] = 0.9
            self.TrajectInfo["bPiping"] = 300
        elif traject_name == "16-3":
            self.TrajectInfo["TrajectLength"] = 19899
            self.TrajectInfo["Pmax"] = 1.0 / 10000
            self.TrajectInfo["omegaPiping"] = 0.24
            self.TrajectInfo["aPiping"] = 0.9
            self.TrajectInfo["bPiping"] = 300
        elif traject_name == "38-1":
            self.TrajectInfo["TrajectLength"] = 28902
            self.TrajectInfo["Pmax"] = 1.0 / 10000
            self.TrajectInfo["omegaPiping"] = 0.24
            self.TrajectInfo["aPiping"] = 0.9
            self.TrajectInfo["bPiping"] = 300

    def read_general_info(self, input_dir: Path, sheet_name: str):
        # Read general data from sheet in standardized xlsx file
        df = pd.read_excel(input_dir.joinpath(self.name + ".xlsx"), sheet_name=None)

        for name, sheet_data in df.items():
            if name == sheet_name:
                data = df[name].set_index(list(df[name])[0])
                self.mechanism_data = {}

                for i in range(len(data)):
                    if (
                        data.index[i] == "Overflow"
                        or data.index[i] == "Piping"
                        or data.index[i] == "StabilityInner"
                    ):
                        self.mechanism_data[data.index[i]] = (
                            data.loc[data.index[i]][0],
                            data.loc[data.index[i]][1],
                        )
                        # setattr(self, data.index[i], (data.loc[data.index[i]][0], data.loc[data.index[i]][1]))
                    else:
                        setattr(self, data.index[i], (data.loc[data.index[i]][0]))
                        # if data.index[i] == 'YearlyWLRise':
                        #     self.YearlyWLRise = self.YearlyWLRise * 3
                        #     print('Warning: WLRise multiplied!')

            elif name == "Housing":
                self.houses = (
                    df["Housing"]
                    .set_index("distancefromtoe")
                    .rename(columns={"number": "cumulative"})
                )
                # self.houses = pd.concat([df["Housing"], pd.DataFrame(np.cumsum(df["Housing"]['number'].values), columns=['cumulative'])], axis=1, join='inner').set_index(
                #     'distancefromtoe')
            else:
                self.houses = None

        # and we add the geometry
        setattr(
            self, "InitialGeometry", df["Geometry"].set_index(list(df["Geometry"])[0])
        )
