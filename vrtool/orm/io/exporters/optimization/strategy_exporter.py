import logging

import numpy as np

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

    def export_dom(self, dom_model: StrategyProtocol) -> None:
        dims = len(dom_model.measures_taken)
        _step_results_section = []
        _step_results_mechanism = []

        _lcc_per_section = {}
        _total_lcc = 0
        for dim_idx in range(0, dims):
            section = dom_model.measures_taken[dim_idx][0]
            _measure_sh_id = dom_model.measures_taken[dim_idx][1] - 1
            _measure_sg_id = dom_model.measures_taken[dim_idx][2] - 1
            _measure_sh = dom_model.sections[section].sh_combinations[_measure_sh_id]
            _measure_sg = dom_model.sections[section].sg_combinations[_measure_sg_id]
            # _aggr_msr = self.find_aggregated(dom_model.sections[section].aggregated_measure_combinations, _measure_sh, _measure_sg)
            _measures = [_measure_sh.primary, _measure_sg.primary]

            # get index of aggregate of primary measure:
            _aggregated_primary = [
                agg_measure
                for agg_measure in dom_model.sections[
                    section
                ].aggregated_measure_combinations
                if agg_measure.check_primary_measure_result_id_and_year(
                    _measure_sh.primary, _measure_sg.primary
                )
            ]
            # select only the first if there are more (the primary measure needs to be identical). We can also add a check TODO
            if any(_aggregated_primary):
                logging.debug(
                    f"More than one aggregated primary measure found for section {dom_model.sections[section].section_name}. Only the first one will be used."
                )
                _aggregated_primary = [_aggregated_primary[0]]

            # get ids of secondary measures
            _secondary_measures = [
                _measure
                for _measure in [_measure_sh.secondary, _measure_sg.secondary]
                if _measure is not None
            ]
            _lcc_per_section[section] = _measure_sh.lcc + _measure_sg.lcc

            # get total_lcc and total_risk values
            _total_lcc = sum(_lcc_per_section.values())
            _total_risk = dom_model.total_risk_per_step[dim_idx + 1]

            for single_measure in _secondary_measures + _aggregated_primary:

                _option_selected_measure_result = (
                    self._get_optimization_selected_measure(
                        single_measure.measure_result_id, single_measure.year
                    )
                )
                _created_optimization_step = OptimizationStep.create(
                    step_number=dim_idx + 1,
                    optimization_selected_measure=_option_selected_measure_result,
                    total_lcc=_total_lcc,
                    total_risk=_total_risk,
                )

                _prob_per_step = dom_model.probabilities_per_step[dim_idx + 1]
                lcc = _measure_sh.lcc + _measure_sg.lcc
                for _t in dom_model._time_periods:
                    _prob_section = self._get_section_time_value(
                        section, _t, _prob_per_step
                    )
                    _step_results_section.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "time": _t,
                            "beta": pf_to_beta(_prob_section),
                            "lcc": lcc,
                        }
                    )

                # From OptimizationSelectedMeasure get measure_result_id based on singleMsrId
                for (
                    _measure_result_mechanism
                ) in (
                    _option_selected_measure_result.measure_result.measure_result_mechanisms
                ):
                    _mechanism_name = MechanismEnum.get_enum(
                        _measure_result_mechanism.mechanism_per_section.mechanism.name
                    ).name
                    _t = _measure_result_mechanism.time
                    _prob_mechanism = self._get_selected_time(
                        section, _t, _mechanism_name, _prob_per_step
                    )
                    _step_results_mechanism.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "mechanism_per_section_id": _measure_result_mechanism.mechanism_per_section_id,
                            "time": _measure_result_mechanism.time,
                            "beta": pf_to_beta(_prob_mechanism),
                            "lcc": lcc,
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
        self, section: int, t: int, values: dict[str, np.ndarray]
    ) -> float:
        pt = 1.0
        for m in values.keys():
            # fix for t=100 where 99 is the last
            maxt = values[m].shape[1] - 1
            _t = min(t, maxt)
            pt *= 1.0 - values[m][section, _t]
        return 1.0 - pt

    def _get_selected_time(
        self,
        section: int,
        t: int,
        mechanism: str,
        values: dict[str, np.ndarray],
    ) -> float:
        if mechanism == "SECTION":
            return self._get_section_time_value(section, t, values)
        maxt = values[mechanism].shape[1] - 1
        _t = min(t, maxt)
        return values[mechanism][section, _t]
