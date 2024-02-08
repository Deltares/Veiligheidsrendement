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
from vrtool.optimization.measures.sg_measure import ALLOWED_MECHANISMS_SG, SgMeasure
from vrtool.optimization.measures.sh_measure import ALLOWED_MECHANISMS_SH, ShMeasure


class StrategyController:
    def __init__(self, method: str, vrtool_config: VrtoolConfig) -> None:
        self._method: str = method
        self._vrtool_config: VrtoolConfig = vrtool_config
        self._section_measures_input: list[SectionAsInput] = []

    def _get_mechanism_year_collection(
        self, measure_data: df, idx: int, allowed_mechanisms: list[MechanismEnum]
    ) -> MechanismPerYearProbabilityCollection:

        def _get_mech_cols(
            mech: MechanismEnum,
            cols: list[MultiIndex],
            allowed_mechanisms: list[MechanismEnum],
        ) -> list[MultiIndex] | None:
            """Get mechanism columns from measure data"""
            if mech in allowed_mechanisms:
                return [col for col in cols if col[0] == mech.name]
            return []

        _probabilities = []
        _cols = measure_data.columns

        for _mech in MechanismEnum:
            _mech_cols = _get_mech_cols(_mech, _cols, allowed_mechanisms)
            for _mech_col in _mech_cols:
                _mech_year = _mech_col[1]
                _beta = measure_data.at[idx, _mech_col]
                _probabilities.append(
                    MechanismPerYear(
                        mechanism=_mech,
                        year=_mech_year,
                        probability=0,  # TODO
                        beta=_beta,
                    )
                )

        return MechanismPerYearProbabilityCollection(_probabilities)

    def _get_measures(self, measure_data: df) -> list[MeasureAsInputProtocol]:
        _measures: list[MeasureAsInputProtocol] = []

        for _idx in measure_data.index:
            _meas_type = MeasureTypeEnum.get_enum(measure_data.at[_idx, ("type", "")])
            _comb_type = CombinableTypeEnum.get_enum(
                measure_data.at[_idx, ("class", "")]
            )
            _cost = measure_data.at[_idx, ("cost", "")]
            _year = measure_data.at[_idx, ("year", "")]
            _lcc = _cost / (1 + self._vrtool_config.discount_rate) ** _year
            _dberm = measure_data.at[_idx, ("dberm", "")]
            _dcrest = measure_data.at[_idx, ("dcrest", "")]

            if _dberm == 0 or _dberm == -999:  # Sg
                _mech_year_coll = self._get_mechanism_year_collection(
                    measure_data,
                    _idx,
                    ALLOWED_MECHANISMS_SH,
                )
                _beta = measure_data.at[_idx, ("beta_target", "")]
                _trans_level = measure_data.at[_idx, ("transition_level", "")]
                _measures.append(
                    ShMeasure(
                        measure_type=_meas_type,
                        combine_type=_comb_type,
                        cost=_cost,
                        year=_year,
                        lcc=_lcc,
                        mechanism_year_collection=_mech_year_coll,
                        beta_target=_beta,
                        transition_level=_trans_level,
                        dcrest=_dcrest,
                    )
                )
            elif _dcrest == 0 or _dcrest == -999:  # Sh
                _mech_year_coll = self._get_mechanism_year_collection(
                    measure_data,
                    _idx,
                    ALLOWED_MECHANISMS_SG,
                )
                _measures.append(
                    SgMeasure(
                        measure_type=_meas_type,
                        combine_type=_comb_type,
                        cost=_cost,
                        year=_year,
                        lcc=_lcc,
                        mechanism_year_collection=_mech_year_coll,
                        dberm=_dberm,
                        dcrest=_dcrest,
                    )
                )
        return _measures

    def map_input(
        self,
        selected_traject: DikeTraject,
        solutions_dict: dict[str, Solutions],
    ) -> None:
        """
        Maps the input to the controller.

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
