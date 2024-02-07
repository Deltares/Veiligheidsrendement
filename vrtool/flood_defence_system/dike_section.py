from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class DikeSection:
    """
    Initialize the DikeSection class, as a general class for a dike section that contains all basic information
    """

    crest_height: float
    mechanism_data: dict[MechanismEnum, tuple[str, str]]
    section_reliability: SectionReliability
    TrajectInfo: DikeTrajectInfo
    name: str
    InitialGeometry: pd.DataFrame
    Length: float
    houses: pd.DataFrame
    with_measures: bool

    def __init__(self) -> None:
        self.mechanism_data = {}
        self.section_reliability = SectionReliability()
        self.TrajectInfo = None
        self.name = ""
        self.Length = float("nan")
        self.crest_height = float("nan")
        self.InitialGeometry = None
        self.houses = None
        self.with_measures = True

    @classmethod
    def get_dike_sections_from_vr_config(
        cls, vrtool_config: VrtoolConfig
    ) -> list[DikeSection]:
        """
        **TODO: DEPRECATED**
        Gets a collection of dike sections based on the config.

        Args:
            vrtool_config (VrtoolConfig): The configuration to retrieve the collection of dike sections with.

        Raises:
            IOError: Raised when no dike sections are found.

        Returns:
            list[DikeSection]: A collection of dike sections.
        """
        files = [i for i in vrtool_config.input_directory.glob("*DV*") if i.is_file()]
        if len(files) == 0:
            raise IOError("Error: no dike sections found. Check path!")

        def get_dike_section(section_filepath: Path) -> DikeSection:
            _dike_section = cls()
            _dike_section.name = section_filepath.stem
            _dike_section.read_general_info(
                section_filepath, "General", vrtool_config.mechanisms
            )
            _dike_section.set_section_reliability(
                vrtool_config.input_directory,
                vrtool_config.mechanisms,
                vrtool_config.T,
                vrtool_config.t_0,
                vrtool_config.externals,
            )
            return _dike_section

        return list(map(get_dike_section, files))

    def read_general_info(
        self, section_filepath: Path, sheet_name: str, available_mechs: list
    ):
        """Reads the general information of a dike section.

        Args:
            section_filepath (Path): The path to read the data from.
            sheet_name (str): The name of the sheet to read the data from.
        """
        # Read general data from sheet in standardized xlsx file
        df = pd.read_excel(section_filepath, sheet_name=None)

        for name, sheet_data in df.items():
            if name == sheet_name:
                data = df[name].set_index(list(df[name])[0])
                self.mechanism_data = {}
                for i in range(len(data)):
                    if data.index[i] in available_mechs:
                        mechanism = MechanismEnum.get_enum(data.index[i])
                        self.mechanism_data[mechanism] = (
                            data.loc[data.index[i]][0],
                            data.loc[data.index[i]][1],
                        )
                        # setattr(self, data.index[i], (data.loc[data.index[i]][0], data.loc[data.index[i]][1]))
                    else:
                        setattr(self, data.index[i], (data.loc[data.index[i]][0]))
                        # if data.index[i] == 'YearlyWLRise':
                        #     self.YearlyWLRise = self.YearlyWLRise * 3
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

    def set_section_reliability(
        self,
        input_path: Path,
        mechanisms: list[MechanismEnum],
        t_value: float,
        t_0: float,
        externals_path: Path = None,
    ):
        """Sets the reliability of the dike section.

        Args:
            input_path (Path): The path to retrieve the input from.
            mechanisms (list[MechanismEnum]): A collection of the mechanisms to consider.
            t_value (float): The year to compute the reliability for.
            t_0 (float): The initial year.
        """
        self.section_reliability = SectionReliability()
        self.section_reliability.load = self._get_load_input(input_path)

        # Then the input for all the mechanisms:
        for mechanism in mechanisms:
            reliability_collection = self._get_mechanism_reliability_collection(
                input_path.joinpath(mechanism.name),
                input_path.joinpath("Stix"),
                externals_path,
                mechanism,
                self.mechanism_data[mechanism],
                t_value,
                t_0,
                self.section_reliability.load.load_type,
            )

            self.section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                reliability_collection
            )

    def _get_load_input(self, input_path: Path) -> LoadInput:
        """
        TODO: Deprecated as we now load from database.
        """
        # Read the data per mechanism, and first the load frequency line:
        _load = LoadInput(list(self.__dict__.keys()))
        if _load.load_type == "HRING":  # 2 HRING computations for different years
            _load.set_HRING_input(
                input_path.joinpath("Waterstand"), self.__dict__
            )  # input folder, location
        elif _load.load_type == "SAFE":  # 2 computation as done for SAFE
            _load.set_fromDesignTable(input_path.joinpath("Toetspeil", self.LoadData))
            _load.set_annual_change(
                change_type="SAFE",
                parameters=[
                    self.YearlyWLRise,
                    self.HBNRise_factor,
                ],
            )
        return _load

    def _get_mechanism_reliability_collection(
        self,
        mechanism_path: Path,
        stix_path: Path,
        externals_path: Path,
        mechanism: MechanismEnum,
        mechanism_data,
        t_value: float,
        t_0: float,
        load_type: str,
    ):
        _mechanism_collection = MechanismReliabilityCollection(
            mechanism, mechanism_data[1], t_value, t_0, 0
        )
        for k in _mechanism_collection.Reliability.keys():
            if load_type == "HRING":
                _mechanism_collection.Reliability[k].Input.fill_mechanism(
                    mechanism_path,
                    stix_path,
                    externals_path,
                    *mechanism_data,
                    mechanism=mechanism,
                    crest_height=self.Kruinhoogte,
                    dcrest=self.Kruindaling,
                )
            else:
                _mechanism_collection.Reliability[k].Input.fill_mechanism(
                    mechanism_path,
                    stix_path,
                    externals_path,
                    *mechanism_data,
                    mechanism=mechanism,
                )
        return _mechanism_collection
