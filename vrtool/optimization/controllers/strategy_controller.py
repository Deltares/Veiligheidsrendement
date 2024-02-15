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
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class StrategyController:
    _method: str
    _vrtool_config: VrtoolConfig
    _section_measures_input: list[SectionAsInput]
    Pf = dict[str, np.ndarray]

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
        _previous_sh_measure: MeasureAsInputProtocol | None = None
        _previous_sg_measure: MeasureAsInputProtocol | None = None

        for _idx in measure_data.index:
            _dberm = measure_data.at[_idx, ("dberm", "")]
            if _dberm == 0 or _dberm == -999:  # Sh
                _sh_measure = self._create_sh_measure(measure_data.iloc[[_idx]], _idx)
                _sh_measure.set_start_cost(_previous_sh_measure)
                _sh_measures.append(_sh_measure)
                _previous_sh_measure = _sh_measure

            _dcrest = measure_data.at[_idx, ("dcrest", "")]
            if _dcrest == 0 or _dcrest == -999:  # Sg
                _sg_measure = self._create_sg_measure(measure_data.iloc[[_idx]], _idx)
                _sg_measure.set_start_cost(_previous_sg_measure)
                _sg_measures.append(_sg_measure)
                _previous_sg_measure = _sg_measure

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
        for _section in self._section_measures_input:
            # initialize the Pf dictionary
            # for _mechanism in ShMeasure.get_allowed_mechanisms():
            #     if _mechanism == 
            #     self.Pf[_mechanism.name] = np.full(
            #         (N, len(_section.sh_measures, _section.max_year))
            #     )
