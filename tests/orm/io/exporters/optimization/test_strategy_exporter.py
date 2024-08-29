import pytest

from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
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

            def __init__(self) -> None:
                self.time_periods = config_t_values
                _sam = []
                for _iy in investment_years:
                    for _idx in [0, 1]:
                        _sam.append((_idx, get_dummy_measure(_iy)))
                self.selected_aggregated_measures = _sam

        return MockedStrategy()

    @pytest.mark.parametrize(
        "config_t",
        [
            pytest.param(
                _basic_config_t_values + [0, 100], id="Including t=0 and t=100"
            ),
            pytest.param(_basic_config_t_values + [100], id="Including t=100"),
            pytest.param(_basic_config_t_values + [0], id="Including t=0"),
            pytest.param(_basic_config_t_values, id="Excluding t=0 and t=100"),
        ],
    )
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
        config_t: list[int],
        investment_years: list[int],
    ):
        # 1. Define test data.
        _expected_values = list(
            sorted(
                set(
                    _basic_config_t_values
                    + [0, 100]
                    + investment_years
                    + [iy - 1 for iy in investment_years]
                )
            )
        )
        _strategy_run = self._get_mocked_strategy_run(config_t, investment_years)

        # 2. Run test.
        _time_periods_to_export = StrategyExporter.get_time_periods_to_export(
            _strategy_run
        )

        # 3. Verify expectations.
        assert _time_periods_to_export == _expected_values

    def test_get_time_periods_to_export_includes_always_0_and_100(
        self,
    ):
        # 1. Define test data.
        _expected_values = list(sorted(_basic_config_t_values + [0, 100]))
        assert 0 not in _basic_config_t_values
        assert 100 not in _basic_config_t_values

        _strategy_run = self._get_mocked_strategy_run(_basic_config_t_values, [])

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
                    _basic_config_t_values
                    + [_bc - 1 for _bc in _basic_config_t_values]
                    + [0, 100]
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
