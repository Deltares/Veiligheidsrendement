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
        _step_lcc_tuples = [
            (
                "step_idx",
                "section_name",
                "aggregation_lcc",
                "accumulated_lcc",
                "last_lcc_values",
            )
        ]
        _accumulated_total_lcc_per_step = []

        for _step_idx, (
            _section_idx,
            _aggregated_measure,
        ) in enumerate(strategy_run.selected_aggregated_measures):
            _accumulated_lcc_last_step = (
                _accumulated_total_lcc_per_step[-1]
                if any(_accumulated_total_lcc_per_step)
                else 0
            )
            _accumulated_total_lcc_per_step.append(
                _aggregated_measure.lcc
                + _accumulated_lcc_last_step
                - _last_step_lcc_per_section[_section_idx]
            )
            _step_total_lcc = _accumulated_total_lcc_per_step[-1]

            # Update the increment dictionary.
            _last_step_lcc_per_section[_section_idx] = _aggregated_measure.lcc

            # get ids of secondary measures
            _secondary_measures = [
                _measure
                for _measure in [
                    _aggregated_measure.sh_combination.secondary,
                    _aggregated_measure.sg_combination.secondary,
                ]
                if _measure is not None
            ]

            _total_risk = strategy_run.total_risk_per_step[_step_idx + 1]
            for single_measure in _secondary_measures + [_aggregated_measure]:
                _step_lcc_tuples.append(
                    (
                        _step_idx + 1,
                        _section_idx,
                        _aggregated_measure.lcc,
                        _accumulated_total_lcc_per_step[-1],
                        str(list(_last_step_lcc_per_section.items())).strip("[]"),
                    )
                )
                _option_selected_measure_result = (
                    self._get_optimization_selected_measure(
                        single_measure.measure_result_id, single_measure.year
                    )
                )
                _created_optimization_step = OptimizationStep.create(
                    step_number=_step_idx + 1,
                    optimization_selected_measure=_option_selected_measure_result,
                    total_lcc=_step_total_lcc,
                    total_risk=_total_risk,
                )

                _prob_per_step = strategy_run.probabilities_per_step[_step_idx + 1]
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

        # from pathlib import Path; _lcc_timeline=list(str(slt).strip("()").replace("'", "\"") for slt in _step_lcc_tuples); Path("lcc_values.csv").write_text("\n".join(_lcc_timeline))
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
