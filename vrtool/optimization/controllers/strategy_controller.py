import numpy as np
import pandas as pd

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.probabilistic_tools.combin_functions import CombinFunctions
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class StrategyController:
    _method: str
    _vrtool_config: VrtoolConfig
    _section_measures_input: list[SectionAsInput]
    # Input variables for optimization:
    opt_parameters: dict[str, int] = {}
    Pf: dict[str, np.ndarray] = {}
    LCCOptions: np.ndarray = np.array([])
    Cint_h: np.ndarray = np.array([])
    Cint_g: np.ndarray = np.array([])
    Dint: np.ndarray = np.array([])
    D: np.ndarray = np.array([])
    RiskGeotechnical: np.ndarray = np.array([])
    RiskOverflow: np.ndarray = np.array([])
    RiskRevetment: np.ndarray = np.array([])

    def __init__(self, method: str, vrtool_config: VrtoolConfig) -> None:
        self._method = method
        self._vrtool_config = vrtool_config
        self._section_measures_input = []

    @staticmethod
    def _get_mechanism_year_collection(
        measure_row: pd.DataFrame,
        idx: int,
        allowed_mechanisms: list[MechanismEnum],
    ) -> MechanismPerYearProbabilityCollection:

        _probabilities = []
        _cols = measure_row.columns

        for _mech in filter(lambda x: x in allowed_mechanisms, MechanismEnum):
            _mech_probs = list(
                map(
                    lambda mech_col: MechanismPerYear(
                        mechanism=_mech,
                        year=mech_col[1],
                        probability=beta_to_pf(measure_row.at[idx, mech_col]),
                    ),
                    filter(lambda x: x[0] == _mech.name, _cols),
                )
            )
            _probabilities.extend(_mech_probs)

        return MechanismPerYearProbabilityCollection(_probabilities)

    def _create_sg_measure(
        self,
        measure_row: pd.DataFrame,
        idx: int,
    ) -> SgMeasure:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_row,
            idx,
            SgMeasure.get_allowed_mechanisms(),
        )
        _meas_type = MeasureTypeEnum.get_enum(measure_row.at[idx, ("type", "")])
        _comb_type = CombinableTypeEnum.get_enum(measure_row.at[idx, ("class", "")])
        _cost = measure_row.at[idx, ("cost", "")]
        _year = measure_row.at[idx, ("year", "")]
        _dberm = measure_row.at[idx, ("dberm", "")]
        _dcrest = measure_row.at[idx, ("dcrest", "")]

        return SgMeasure(
            measure_type=_meas_type,
            combine_type=_comb_type,
            cost=_cost,
            year=_year,
            discount_rate=self._vrtool_config.discount_rate,
            mechanism_year_collection=_mech_year_coll,
            dberm=_dberm,
            dcrest=_dcrest,
        )

    def _create_sh_measure(
        self,
        measure_row: pd.DataFrame,
        idx: int,
    ) -> ShMeasure:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_row,
            idx,
            ShMeasure.get_allowed_mechanisms(),
        )
        _meas_type = MeasureTypeEnum.get_enum(measure_row.at[idx, ("type", "")])
        _comb_type = CombinableTypeEnum.get_enum(measure_row.at[idx, ("class", "")])
        _cost = measure_row.at[idx, ("cost", "")]
        _year = measure_row.at[idx, ("year", "")]
        _dcrest = measure_row.at[idx, ("dcrest", "")]
        _beta = measure_row.at[idx, ("beta_target", "")]
        _trans_level = measure_row.at[idx, ("transition_level", "")]

        return ShMeasure(
            measure_type=_meas_type,
            combine_type=_comb_type,
            cost=_cost,
            year=_year,
            discount_rate=self._vrtool_config.discount_rate,
            mechanism_year_collection=_mech_year_coll,
            beta_target=_beta,
            transition_level=_trans_level,
            dcrest=_dcrest,
        )

    def _get_measures(self, measure_data: pd.DataFrame) -> list[MeasureAsInputProtocol]:
        # Note: this method requires the order of measures is such that the "0-measure" comes first
        _sh_measures: list[MeasureAsInputProtocol] = []
        _sg_measures: list[MeasureAsInputProtocol] = []

        for _idx in measure_data.index:
            _dberm = measure_data.at[_idx, ("dberm", "")]
            if _dberm == 0 or _dberm == -999:  # Sh
                _sh_measure = self._create_sh_measure(measure_data.iloc[[_idx]], _idx)
                _sh_measure.set_start_cost(_sh_measures[-1] if _sh_measures else None)
                _sh_measures.append(_sh_measure)

            _dcrest = measure_data.at[_idx, ("dcrest", "")]
            if _dcrest == 0 or _dcrest == -999:  # Sg
                _sg_measure = self._create_sg_measure(measure_data.iloc[[_idx]], _idx)
                _sg_measure.set_start_cost(_sg_measures[-1] if _sg_measures else None)
                _sg_measures.append(_sg_measure)

        return _sh_measures + _sg_measures

    def map_input(
        self,
        selected_traject: DikeTraject,
        solutions_dict: dict[str, Solutions],
    ) -> None:
        """
        Maps the dataframe input to the controller (temporarily).

        Args:
            selected_traject (DikeTraject): Selected dike traject.
            solutions_dict (dict[str, Solutions]): Solutions dictionary.
        """
        for _section in selected_traject.sections:
            _section_name = _section.name
            self._section_measures_input.append(
                SectionAsInput(
                    _section_name,
                    selected_traject.general_info.traject_name,
                    selected_traject.general_info.FloodDamage,
                    self._get_measures(solutions_dict[_section_name].MeasureData),
                )
            )

    def combine(self) -> None:
        """
        Combines the measures for each section.
        """
        for _section in self._section_measures_input:
            _combine_controller = CombineMeasuresController(_section)
            _section.combined_measures = _combine_controller.combine()

    def aggregate(self) -> None:
        """
        Aggregates combinations of measures for each section.
        """
        for _section in self._section_measures_input:
            _aggregate_controller = AggregateCombinationsController(_section)
            _section.aggregated_measure_combinations = _aggregate_controller.aggregate()

    def map_output(self) -> None:
        """
        Maps the aggregate combinations of measures to the legacy output (temporarily).
        """
        # Initialize datastructures
        mechanisms = set(
            mech for sect in self._section_measures_input for mech in sect.mechanisms
        )

        def _init_section_structures(
            sections: list[SectionAsInput], mechanisms: set[MechanismEnum]
        ) -> tuple[dict[str, int], dict[str, np.array], np.array]:
            parameters: dict[str, int] = {}
            pf: dict[str, np.ndarray] = {}
            lcc: np.ndarray = np.array([])

            _num_sections = len(sections)
            _max_year = max(s.max_year for s in sections)
            _max_sg = max(map(len, (s.sg_combinations for s in sections)))
            _max_sh = max(map(len, (s.sh_combinations for s in sections)))

            # General parameters
            parameters = {
                "N": _num_sections,
                "T": _max_year,
                "Sg": _max_sg + 1,
                "Sh": _max_sh + 1,
            }

            # Probabilities
            for _mech in mechanisms:
                if _mech == MechanismEnum.OVERFLOW:
                    pf[_mech.name] = np.full(
                        (
                            _num_sections,
                            _max_sh + 1,
                            _max_year,
                        ),
                        1.0,
                    )
                elif _mech == MechanismEnum.REVETMENT:
                    pf[_mech.name] = np.full(
                        (
                            _num_sections,
                            _max_sh + 1,
                            _max_year,
                        ),
                        1.0e-18,
                    )
                else:
                    pf[_mech.name] = np.full(
                        (
                            _num_sections,
                            _max_sg + 1,
                            _max_year,
                        ),
                        1.0,
                    )

            # LCC
            lcc = np.full((_num_sections, _max_sh + 1, _max_sg + 1), 1e99)

            return (parameters, pf, lcc)

        (self.opt_parameters, self.Pf, self.LCCOptions) = _init_section_structures(
            self._section_measures_input, mechanisms
        )

        def _get_combination_idx(
            comb: CombinedMeasure, combinations: list[CombinedMeasure]
        ) -> int:
            """
            Find the index of the combination in the list of combinations of measures.

            Args:
                comb (CombinedMeasure): The combination at hand.
                combinations (list[CombinedMeasure]): LIs of all combined measures.

            Returns:
                int: Index of the combined measures in the list.
            """
            return next((i for i, c in enumerate(combinations) if c == comb), -1)

        def _get_pf_for_measures(
            mech: MechanismEnum,
            combinations: list[CombinedMeasure],
            dims: tuple[int, ...],
        ) -> np.ndarray:
            _probs = np.zeros(dims)
            # Add other measures
            for m, _meas in enumerate(combinations):
                _probs[m + 1, :] = _meas.mechanism_year_collection.get_probabilities(
                    mech, list(range(self.opt_parameters["T"]))
                )
            return _probs

        def _get_pf_for_mech(
            mech: MechanismEnum, section: SectionAsInput, dims: tuple[int, ...]
        ) -> np.ndarray:
            # Get initial assessment as first measure
            _initial_probs = (
                section.initial_assessment.mechanism_year_collection.get_probabilities(
                    mech, list(range(self.opt_parameters["T"]))
                )
            )
            # Get probabilities for all measures
            if section.sg_measures[0].is_mechanism_allowed(mech):
                return np.concatenate(
                    _initial_probs,
                    _get_pf_for_measures(
                        mech, section.sg_combinations, (dims[0], dims[1] - 1), axis=0
                    ),
                )
            elif section.sh_measures[0].is_mechanism_allowed(mech):
                return _get_pf_for_measures(mech, section.sh_combinations, dims)

            raise ValueError("Mechanism not allowed")

        # Populate datastructure per section, per mechanism, per sg/sh measure, per year
        for n, _section in enumerate(self._section_measures_input):
            # Probabilities
            for _mech in mechanisms:
                _pf = _get_pf_for_mech(
                    _mech,
                    _section,
                    self.Pf[_mech.name].shape[1:],
                )
                self.Pf[_mech.name][n, 0 : len(_pf), :] = _pf

            # LCC
            for _aggr in _section.aggregated_measure_combinations:
                _sh_idx = _get_combination_idx(
                    _aggr.sh_combination, _section.sh_combinations
                )
                _sg_idx = _get_combination_idx(
                    _aggr.sg_combination, _section.sg_combinations
                )
                self.LCCOptions[n, _sh_idx, _sg_idx] = _aggr.lcc

        # Initialize/calculate decision variables
        # - for executed options [N, Sh] & [N, Sg]
        self.Cint_h = np.zeros(
            (self.opt_parameters["N"], self.opt_parameters["Sh"] - 1)
        )
        self.Cint_g = np.zeros(
            (self.opt_parameters["N"], self.opt_parameters["Sg"] - 1)
        )
        # - for weakest overflow section with dims [N,Sh]
        self.Dint = np.zeros((self.opt_parameters["N"], self.opt_parameters["Sh"] - 1))
        # - for discounted damage [T,]
        self.D = np.array(
            self._section_measures_input[0].flood_damage
            * (
                1
                / (
                    (1 + self._section_measures_input[0].measures[0].discount_rate)
                    ** np.arange(0, self.opt_parameters["T"], 1)
                )
            )
        )

        # Calculate expected damage
        # - for overflow//piping/slope stability
        def _get_independent_probability_of_failure(
            probability_of_failure_lookup: dict[str, np.array]
        ) -> np.array:
            return CombinFunctions.combine_probabilities(
                probability_of_failure_lookup,
                SgMeasure.get_allowed_mechanisms(),
            )

        self.RiskGeotechnical = _get_independent_probability_of_failure(
            self.Pf
        ) * np.tile(self.D.T, (self.opt_parameters["N"], self.opt_parameters["Sg"], 1))

        self.RiskOverflow = self.Pf[MechanismEnum.OVERFLOW.name] * np.tile(
            self.D.T, (self.opt_parameters["N"], self.opt_parameters["Sh"], 1)
        )

        # - for revetment
        self.RiskRevetment = []
        if MechanismEnum.REVETMENT in mechanisms:
            self.RiskRevetment = self.Pf[MechanismEnum.REVETMENT.name] * np.tile(
                self.D.T, (self.opt_parameters["N"], self.opt_parameters["Sh"], 1)
            )
        else:
            self.RiskRevetment = np.zeros(
                (
                    self.opt_parameters["N"],
                    self.opt_parameters["Sh"],
                    self.opt_parameters["T"],
                )
            )
