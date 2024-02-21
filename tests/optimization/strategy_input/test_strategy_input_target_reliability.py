from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)
from vrtool.optimization.strategy_input.strategy_input_target_reliability import (
    StrategyInputTargetReliability,
)


class TestStrategyInputTargetReliability:
    def test_initialize(self):
        # 1. Run test.
        _strategy_input = StrategyInputTargetReliability()

        # 2. Validate expectations.
        assert isinstance(_strategy_input, StrategyInputProtocol)
        assert _strategy_input.design_method == ""
