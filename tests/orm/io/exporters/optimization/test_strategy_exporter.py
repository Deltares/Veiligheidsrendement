import pytest

from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategies.strategy_step import StrategyStep
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.orm.io.exporters.optimization.strategy_exporter import StrategyExporter

_basic_config_t_values = [10, 25]


class TestStrategyExporter:
    def _get_mocked_strategy_run(
        self, config_t_values: list[int], investment_years: list[int]
    ) -> StrategyProtocol:
        def get_dummy_measure(investment_year: int) -> AggregatedMeasureCombination:
            return AggregatedMeasureCombination(
                sh_combination=None,
                sg_combination=None,
                measure_result_id=-1,
                year=investment_year,
            )

        class MockedStrategy(StrategyProtocol):
            """
            Mocked strategy just to generate time periods and
            aggregated measures with investment years to test
            the correct generation of years to export.
            """
            optimization_steps: list[StrategyStep] = []

            def __init__(self) -> None:
                self.time_periods = config_t_values                
                for _iy in investment_years:
                    for _idx in [0, 1]:
                        self.optimization_steps.append(StrategyStep(
                            section_idx=_idx,
                            aggregated_measure=get_dummy_measure(_iy)
                        ))

        return MockedStrategy()

    @pytest.mark.parametrize(
        "investment_years",
        [
            pytest.param(
                _basic_config_t_values, id="With investment years in config_t"
            ),
            pytest.param(
                [bcv - 1 for bcv in _basic_config_t_values],
                id="With investment years (-1) in config_t",
            ),
            pytest.param(
                [3, 23],
                id="With unrelated values to config_t",
            ),
        ],
    )
    def test_get_time_periods_to_export(
        self,
        investment_years: list[int],
    ):
        # 1. Define test data.
        _expected_values = list(
            sorted(
                set(
                    _basic_config_t_values
                    + investment_years
                    + [iy - 1 for iy in investment_years]
                )
            )
        )
        _strategy_run = self._get_mocked_strategy_run(
            _basic_config_t_values, investment_years
        )

        # 2. Run test.
        _time_periods_to_export = StrategyExporter.get_time_periods_to_export(
            _strategy_run
        )

        # 3. Verify expectations.
        assert _time_periods_to_export == _expected_values

    def test_get_time_periods_to_export_includes_previous_investment_year(self):
        # 1. Define test data.
        _expected_values = sorted(
            list(
                set(
                    _basic_config_t_values + [_bc - 1 for _bc in _basic_config_t_values]
                )
            )
        )
        _strategy_run = self._get_mocked_strategy_run(
            _basic_config_t_values, _basic_config_t_values
        )

        # 2. Run test.
        _time_periods_to_export = StrategyExporter.get_time_periods_to_export(
            _strategy_run
        )

        # 3. Verify expectations.
        assert _time_periods_to_export == _expected_values
