from typing import Callable, Iterator

import pytest

from vrtool.defaults.vrtool_config import VrtoolConfig
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
    @pytest.fixture(name="optimization_input_measure_builder_fixture")
    def _get_optimization_input_measure_builder_fixture(
        self,
    ) -> Iterator[Callable[[MeasureAsInputProtocol], OptimizationInputMeasures]]:
        def build_optimization_input_measures(
            mocked_measure: MeasureAsInputProtocol,
        ) -> OptimizationInputMeasures:
            return OptimizationInputMeasures(
                vr_config=VrtoolConfig(),
                selected_traject=456,
                section_input_collection=[
                    SectionAsInput(
                        section_name="asdf",
                        traject_name="456",
                        measures=[mocked_measure],
                        flood_damage=4.2,
                        section_length=42,
                        a_section_piping=2.4,
                        a_section_stability_inner=4.2,
                    )
                ],
            )

        yield build_optimization_input_measures

    def test_init_with_valid_data(
        self,
        optimization_input_measure_builder_fixture: Callable[
            [MeasureAsInputProtocol], OptimizationInputMeasures
        ],
    ):
        # 1. Define test data
        _mocked_measure = MockedMeasure()
        _opt_input_measures = optimization_input_measure_builder_fixture(
            _mocked_measure
        )
        assert isinstance(_opt_input_measures, OptimizationInputMeasures)
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

    def test_given_invalid_vrtool_config_when_initialize_raises(
        self,
        invalid_vrtool_config_fixture: tuple[VrtoolConfig, str],
        optimization_input_measure_builder_fixture: Callable[
            [MeasureAsInputProtocol], OptimizationInputMeasures
        ],
    ):
        # 1. Define test data.
        _invalid_vrtool_config, _expected_error_mssg = invalid_vrtool_config_fixture
        assert isinstance(_invalid_vrtool_config, VrtoolConfig)

        _mocked_measure = MockedMeasure()
        _opt_input_measures = optimization_input_measure_builder_fixture(
            _mocked_measure
        )
        assert isinstance(_opt_input_measures, OptimizationInputMeasures)
        _opt_input_measures.vr_config = _invalid_vrtool_config

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            RunOptimization(_opt_input_measures, None)

        # 3. Verify expectation
        assert str(exc_err.value) == _expected_error_mssg
