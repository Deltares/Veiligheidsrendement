import pytest

from vrtool.optimization.controllers.strategy_controller import StrategyController
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.run_workflows.optimization_workflow.optimization_input_measures import (
    OptimizationInputMeasures,
)
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class MockedMeasure(MeasureAsInputProtocol):
    measure_result_id = 42
    measure_type = None
    combine_type = None
    cost = float("nan")
    discount_rate = float("nan")
    year = 0
    mechanism_year_collection = None
    base_cost = float("nan")


class TestRunOptimization:
    def test_init_with_valid_data(self):
        # 1. Define test data
        _mocked_measure = MockedMeasure()
        _opt_input_measures = OptimizationInputMeasures(
            vr_config="sth",
            selected_traject=456,
            section_input_collection=[
                SectionAsInput(
                    section_name="asdf",
                    traject_name="456",
                    measures=[_mocked_measure],
                    flood_damage=4.2,
                )
            ],
        )
        _optimization_selected_measure_ids = {1: [1], 2: [2]}

        # 2. Run test.
        _run = RunOptimization(_opt_input_measures, _optimization_selected_measure_ids)

        # 3. Verify expectations.
        assert isinstance(_run, RunOptimization)
        assert isinstance(_run, VrToolRunProtocol)
        assert _run.vr_config == _opt_input_measures.vr_config
        assert _run.selected_traject == _opt_input_measures.selected_traject
        assert isinstance(_run._strategy_controller, StrategyController)
        assert _run._selected_measure_ids == _optimization_selected_measure_ids
        assert _run._ids_to_import == [
            (_mocked_measure.measure_result_id, _mocked_measure.year)
        ]

    def test_init_with_invalid_data(self):
        _optimization_selected_measure_ids = {1: [1], 2: [2]}
        with pytest.raises(ValueError) as exception_error:
            RunOptimization("not a result instance", _optimization_selected_measure_ids)

        assert (
            str(exception_error.value)
            == "Required valid instance of OptimizationInputMeasures as an argument."
        )
