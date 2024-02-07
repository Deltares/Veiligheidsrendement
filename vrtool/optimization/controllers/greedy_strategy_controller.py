from pandas import DataFrame as df
from pandas import MultiIndex

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
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


class GreedyStrategyController:
    def __init__(self, method: str, vrtool_config: VrtoolConfig) -> None:
        self._method: str = method
        self._vrtool_config: VrtoolConfig = vrtool_config
        self._section_measures_input: list[SectionAsInput] = []

    def _get_mechanism_year_collection(
        self, measure_data: df, idx: int
    ) -> MechanismPerYearProbabilityCollection:

        def _get_mech_cols(
            mech: MechanismEnum, cols: list[MultiIndex]
        ) -> list[MultiIndex]:
            """Get mechanism columns from measure data"""
            return [col for col in cols if col[0] == mech.name]

        _probabilities = []
        _cols = measure_data.columns
        for _mech in MechanismEnum:
            _mech_cols = _get_mech_cols(_mech, _cols)
            for _mech_col in _mech_cols:
                _mech_year = _mech_col[1]
                _prob = measure_data.at[
                    idx, _mech_col
                ]  # TODO: convert beta to probability
                _probabilities.append(
                    MechanismPerYear(
                        mechanism=_mech,
                        year=_mech_year,
                        probability=_prob,
                    )
                )

        return MechanismPerYearProbabilityCollection(_probabilities)

    def _get_measures(self, measure_data: df) -> list[MeasureAsInputProtocol]:
        _measures = []
        for _idx in measure_data.index:
            _meas_type = MeasureTypeEnum.get_enum(measure_data.at[_idx, ("type", "")])
            _comb_type = CombinableTypeEnum.get_enum(
                measure_data.at[_idx, ("class", "")]
            )
            _cost = measure_data.at[_idx, ("cost", "")]
            _year = measure_data.at[_idx, ("year", "")]
            _lcc = 0  # TODO
            _mech_year_coll = self._get_mechanism_year_collection(measure_data, _idx)
            _measures.append(
                SgMeasure(
                    measure_type=_meas_type,
                    combine_type=_comb_type,
                    cost=_cost,
                    year=_year,
                    lcc=_lcc,
                    mechanism_year_collection=_mech_year_coll,
                    dberm=0,  # TODO
                    dcrest=0,  # TODO
                )
            )
        return _measures

    def map_input(
        self,
        selected_traject: DikeTraject,
        ids_to_import: list[tuple[int, int]],
        optimization_selected_measure_ids: dict[int, list[int]],
        solutions_dict: dict[str, Solutions],
    ) -> None:
        """
        Maps the input to the controller.

        Args:
            selected_traject (DikeTraject): Selected dike traject.
            ids_to_import (list[tuple[int, int]]): List of measure results ids to import.
            optimization_selected_measure_ids (list[tuple[int, int]]):
                List of optimization selected measure ids.
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
