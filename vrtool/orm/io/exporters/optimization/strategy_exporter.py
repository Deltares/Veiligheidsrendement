from collections import defaultdict

import numpy as np

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.optimization import (
    OptimizationStep,
    OptimizationStepResultMechanism,
    OptimizationStepResultSection,
)
from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class ExporterLccCostCalculator:
    """
    Disclaimer, this calculator is not expected to give valid results.
    It is just kept now for research purposes.
    """

    def __init__(self) -> None:
        self._lcc_per_section = {}
        self._initial_legacy_lcc_per_section = {}
        self._total_lcc = 0
        self._soil_reinforcement_paid = False
        self._computed_measure_types = {}

    @property
    def accumulated_lcc(self) -> float:
        """
        Returns:
            float: the accumulated lcc per section
        """
        return sum(self._lcc_per_section.values())

    def add_aggregation(
        self, section: int, aggregated_measure_combination: AggregatedMeasureCombination
    ) -> float:
        """
        Includes the aggregated measure combination to compute the lcc throughout all
        sections.

        Args:
            section (int): Section id for creating the lcc per section catalog.
            aggregated_measure_combination (AggregatedMeasureCombination): Aggregation to i nclude.
        """
        # This is very dirty, but can't come up with a better solution at the moment
        _sh_combination = aggregated_measure_combination.sh_combination
        _sg_combination = aggregated_measure_combination.sg_combination
        _lcc_value = _sh_combination.cost + _sg_combination.cost
        if not self._soil_reinforcement_paid:
            self._soil_reinforcement_paid = (
                _sh_combination.primary.measure_type
                == MeasureTypeEnum.SOIL_REINFORCEMENT
            )

        if section not in self._lcc_per_section:
            # Initial values have not yet been set.
            self._initial_legacy_lcc_per_section[section] = _lcc_value
            if _sh_combination.primary.measure_type not in self._computed_measure_types:
                self._computed_measure_types[
                    _sh_combination.primary.measure_type
                ] = section
                _lcc_value = aggregated_measure_combination.lcc
            else:
                _section_with_computed_measure = self._computed_measure_types[
                    _sh_combination.primary.measure_type
                ]
                self._lcc_per_section[
                    _section_with_computed_measure
                ] = self._initial_legacy_lcc_per_section[_section_with_computed_measure]
        elif any(self._initial_legacy_lcc_per_section):
            # We have already set all initial values, so replace them
            for (
                _initial_section,
                _values,
            ) in self._initial_legacy_lcc_per_section.items():
                self._lcc_per_section[_initial_section] = _values
            self._initial_legacy_lcc_per_section = []

        self._lcc_per_section[section] = _lcc_value


class StrategyExporter(OrmExporterProtocol):
    def __init__(self, optimization_run_id: int) -> None:
        self.optimization_run: OptimizationRun = OptimizationRun.get_by_id(
            optimization_run_id
        )

    def find_aggregated(
        self, combinations: list[AggregatedMeasureCombination], measure_sh, measure_sg
    ):
        for a in combinations:
            if a.sg_combination == measure_sg and a.sh_combination == measure_sh:
                return a

    def export_dom(self, strategy_run: StrategyProtocol) -> None:
        _step_results_section = []
        _step_results_mechanism = []
        _last_step_lcc_per_section = defaultdict(lambda: 0)

        # _accumulated_total_lcc stores always the calculated `lcc` of the previous step
        _accumulated_total_lcc = 0

        for _measure_idx, (
            _section_idx,
            _aggregated_measure,
        ) in enumerate(strategy_run.selected_aggregated_measures):
            _selected_section = strategy_run.sections[_section_idx]

            _accumulated_total_lcc += (
                _aggregated_measure.lcc
                - _last_step_lcc_per_section[_selected_section.section_name]
            )
            # Update the increment dictionary.
            _last_step_lcc_per_section[
                _selected_section.section_name
            ] = _aggregated_measure.lcc

            # get ids of secondary measures
            _secondary_measures = [
                _measure
                for _measure in [
                    _aggregated_measure.sh_combination,
                    _aggregated_measure.sg_combination,
                ]
                if _measure is not None
            ]

            _total_risk = strategy_run.total_risk_per_step[_measure_idx + 1]
            for single_measure in _secondary_measures + [_aggregated_measure]:

                _option_selected_measure_result = (
                    self._get_optimization_selected_measure(
                        single_measure.measure_result_id, single_measure.year
                    )
                )
                _created_optimization_step = OptimizationStep.create(
                    step_number=_measure_idx + 1,
                    optimization_selected_measure=_option_selected_measure_result,
                    total_lcc=_accumulated_total_lcc,
                    total_risk=_total_risk,
                )

                _prob_per_step = strategy_run.probabilities_per_step[_measure_idx + 1]
                for _t in strategy_run.time_periods:
                    _prob_section = self._get_section_time_value(
                        _section_idx, _t, _prob_per_step
                    )
                    _step_results_section.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "time": _t,
                            "beta": pf_to_beta(_prob_section),
                            "lcc": _aggregated_measure.lcc,
                        }
                    )

                # From OptimizationSelectedMeasure get measure_result_id based on singleMsrId
                for (
                    _measure_result_mechanism
                ) in (
                    _option_selected_measure_result.measure_result.measure_result_mechanisms
                ):
                    _mechanism = MechanismEnum.get_enum(
                        _measure_result_mechanism.mechanism_per_section.mechanism.name
                    )
                    _t = _measure_result_mechanism.time
                    _prob_mechanism = self._get_selected_time(
                        _section_idx, _t, _mechanism, _prob_per_step
                    )
                    _step_results_mechanism.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "mechanism_per_section_id": _measure_result_mechanism.mechanism_per_section_id,
                            "time": _measure_result_mechanism.time,
                            "beta": pf_to_beta(_prob_mechanism),
                            "lcc": _aggregated_measure.lcc,
                        }
                    )

        OptimizationStepResultSection.insert_many(_step_results_section).execute()
        OptimizationStepResultMechanism.insert_many(_step_results_mechanism).execute()

    def _find_id_in_section(self, measure_id: int, index_section: list[int]) -> int:
        for i in range(len(index_section)):
            if index_section[i][0] == measure_id:
                return i
        raise ValueError(
            "Measure ID {} not found in any of the section indices.".format(measure_id)
        )

    def _get_optimization_selected_measure(
        self, single_msr_id: int, investment_year: int
    ) -> OptimizationSelectedMeasure:
        _opt_selected_measure = (
            self.optimization_run.optimization_run_measure_results.where(
                (OptimizationSelectedMeasure.measure_result_id == single_msr_id)
                & (OptimizationSelectedMeasure.investment_year == investment_year)
            ).get_or_none()
        )
        if not _opt_selected_measure:
            raise ValueError(
                "OptimizationSelectedMeasure with run_id {} and measure result id {} not found".format(
                    self.optimization_run.get_id(), single_msr_id
                )
            )
        return _opt_selected_measure

    def _get_section_time_value(
        self, section: int, t: int, values: dict[MechanismEnum, np.ndarray]
    ) -> float:
        pt = 1.0
        for m in values:
            # fix for t=100 where 99 is the last
            maxt = values[m].shape[1] - 1
            _t = min(t, maxt)
            pt *= 1.0 - values[m][section, _t]
        return 1.0 - pt

    def _get_selected_time(
        self,
        section: int,
        t: int,
        mechanism: MechanismEnum,
        values: dict[MechanismEnum, np.ndarray],
    ) -> float:
        maxt = values[mechanism].shape[1] - 1
        _t = min(t, maxt)
        return values[mechanism][section, _t]
