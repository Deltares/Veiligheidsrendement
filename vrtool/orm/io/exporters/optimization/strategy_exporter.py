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

    @staticmethod
    def get_time_periods_to_export(strategy_run: StrategyProtocol) -> list[int]:
        """
        Gets the list of time periods to export by combining the expected required ones
        ([0, 100]), the time periods defined in the configuration (`VrtoolConfig.T`)
        and the provided investment years (including `investment_year - 1`
        if `investment_year > 1`) for optimization calculation.
        Note: this method is created in relation to VRTOOL-577.

        Args:
            strategy_run (StrategyProtocol): Strategy object containing all required data.

        Returns:
            list[int]: Years whose betas needs to be exported to the database.
        """

        def get_investment_years() -> list[int]:
            _investment_years = []
            for am in (x.aggregated_measure for x in strategy_run.optimization_steps):
                if am.year not in _investment_years:
                    _investment_years.append(am.year)
                _previous_year = am.year - 1
                if _previous_year not in _investment_years and _previous_year > 0:
                    _investment_years.append(_previous_year)
            return _investment_years

        return sorted(list(set(strategy_run.time_periods + get_investment_years())))

    def export_dom(self, strategy_run: StrategyProtocol) -> None:
        def get_step_results_mechanism(mechanism: tuple[int, MechanismEnum]) -> dict:
            _prob_mechanism = self._get_selected_time(
                _optimization_step.section_idx,
                _t,
                mechanism[1],
                _optimization_step.probabilities,
            )
            return {
                "optimization_step": _created_optimization_step,
                "mechanism_per_section_id": mechanism[0],
                "time": _t,
                "beta": pf_to_beta(_prob_mechanism),
                "lcc": _optimization_step.aggregated_measure.lcc,
            }

        _step_results_section = []
        _step_results_mechanism = []

        _time_periods_to_export = self.get_time_periods_to_export(strategy_run)

        for _optimization_step in strategy_run.optimization_steps:

            # get ids of secondary measures
            _secondary_measures = [
                _measure
                for _measure in [
                    _optimization_step.aggregated_measure.sh_combination.secondary,
                    _optimization_step.aggregated_measure.sg_combination.secondary,
                ]
                if _measure is not None
            ]

            for single_measure in _secondary_measures + [
                _optimization_step.aggregated_measure
            ]:
                _opt_selected_measure_result = self._get_optimization_selected_measure(
                    single_measure.measure_result_id, single_measure.year
                )
                _created_optimization_step = OptimizationStep.create(
                    step_number=_optimization_step.step_number,
                    step_type=_optimization_step.step_type.name,
                    optimization_selected_measure=_opt_selected_measure_result,
                    total_lcc=_optimization_step.total_cost,
                    total_risk=_optimization_step.total_risk,
                )

                # Loop over time periods to export including the ones not in the configuration
                _mechs = self._get_mechanisms(_opt_selected_measure_result)
                for _t in _time_periods_to_export:
                    _prob_section = self._get_section_time_value(
                        _optimization_step.section_idx,
                        _t,
                        _optimization_step.probabilities,
                    )

                    # Export section results for time periods
                    _step_results_section.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "time": _t,
                            "beta": pf_to_beta(_prob_section),
                            "lcc": _optimization_step.aggregated_measure.lcc,
                        }
                    )

                    _step_results_mechanism.extend(
                        list(
                            map(
                                get_step_results_mechanism,
                                _mechs,
                            )
                        )
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
        _optimization_selected_measure = (
            self.optimization_run.optimization_run_measure_results.where(
                (OptimizationSelectedMeasure.measure_result_id == single_msr_id)
                & (OptimizationSelectedMeasure.investment_year == investment_year)
            ).get_or_none()
        )
        if not _optimization_selected_measure:
            raise ValueError(
                "OptimizationSelectedMeasure with run_id {} and measure result id {} not found".format(
                    self.optimization_run.get_id(), single_msr_id
                )
            )
        return _optimization_selected_measure

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

    def _get_mechanisms(
        self,
        opt_selected_measure: OptimizationSelectedMeasure,
    ) -> list[tuple[int, MechanismEnum]]:
        return sorted(
            set(
                (
                    m.mechanism_per_section_id,
                    MechanismEnum.get_enum(m.mechanism_per_section.mechanism.name),
                )
                for m in opt_selected_measure.measure_result.measure_result_mechanisms
            )
        )
