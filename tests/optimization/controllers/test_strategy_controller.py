from vrtool.optimization.controllers.strategy_controller import StrategyController


class TestStrategyController:
    def test_strategy_controller_init(self):
        # 1. Define test data.
        _method = ""
        _vrtool_config = None

        # 2. Run test.
        _controller = StrategyController(_method, _vrtool_config)

        # 3. Verify expectations.
        assert isinstance(_controller, StrategyController)
