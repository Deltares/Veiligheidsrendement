from vrtool.optimization.measures.section_as_input import SectionAsInput
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
        assert not any(_strategy_input.section_as_input_dict)

    def test_given_valid_input_when_mapped_then_sets_dict(self):
        # 1. Define test data.
        _input_collection = [
            SectionAsInput(
                section_name="dummy", traject_name="asdf", flood_damage=4.2, measures=[]
            ),
            SectionAsInput(
                section_name="dummy_two",
                traject_name="asdf",
                flood_damage=4.2,
                measures=[],
            ),
        ]

        # 2. Run test.
        _strategy_input = (
            StrategyInputTargetReliability.from_section_as_input_collection(
                _input_collection
            )
        )

        # 3. Verify expectations.
        assert isinstance(_strategy_input, StrategyInputProtocol)
        assert isinstance(_strategy_input, StrategyInputTargetReliability)
        assert isinstance(_strategy_input.section_as_input_dict, dict)
        for _section_as_input in _input_collection:
            assert (
                _section_as_input.section_name
                in _strategy_input.section_as_input_dict.keys()
            )
            assert (
                _strategy_input.section_as_input_dict[_section_as_input.section_name]
                == _section_as_input
            )
