from vrtool.optimization.controllers.strategy_controller import StrategyController


class TestStrategyController:
    def test_strategy_controller_init(self):
        # 1. Define test data.
        _list_section_as_input = []

        # 2. Run test.
        _controller = StrategyController(_list_section_as_input)

        # 3. Verify expectations.
        assert isinstance(_controller, StrategyController)
