from typing import Any, Iterator

import pytest

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.course_sand_barrier_measure_calculator import (
    CourseSandBarrierMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.heavescreen_measure_calculator import (
    HeavescreenMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_geotextile_measure_calculator import (
    VerticalGeotextileMeasureCalculator,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_factory import (
    VerticalPipingMeasureCalculatorFactory,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection


class TestVerticalPipingMeasureCalculatorFactory:
    @pytest.fixture(name="calculator_d_cover_args")
    def _get_calculator_d_cover_args_fixture(
        self, request: pytest.FixtureRequest
    ) -> Iterator[tuple[DikeTrajectInfo, DikeSection, MeasureProtocol]]:
        class MockedMeasure(MeasureProtocol):
            """
             Define required args for calculator initialization.
            Their values do not matter at the context of this test module.
            """

            def __init__(self) -> None:
                self.parameters = dict(year=42)
                self.config = VrtoolConfig()

        _dike_traject = DikeTrajectInfo(traject_name="DummyTraject")
        _dike_section = DikeSection()
        _dike_section.TrajectInfo = _dike_traject
        _dike_section.cover_layer_thickness = request.param

        yield _dike_traject, _dike_section, MockedMeasure()

    @pytest.mark.parametrize(
        "calculator_d_cover_args, expected_calculator_type",
        [
            pytest.param(0, CourseSandBarrierMeasureCalculator, id="d_crest = 0"),
            pytest.param(1.5, CourseSandBarrierMeasureCalculator, id="d_crest < 2"),
            pytest.param(2, VerticalGeotextileMeasureCalculator, id="d_crest == 2"),
            pytest.param(
                3.5, VerticalGeotextileMeasureCalculator, id="2 <= d_crest < 4"
            ),
            pytest.param(4, HeavescreenMeasureCalculator, id="d_crest >= 4"),
            pytest.param(42, HeavescreenMeasureCalculator, id="d_crest > 4"),
        ],
        indirect=["calculator_d_cover_args"],
    )
    def test_get_calculator_given_valid_d_cover_value(
        self,
        calculator_d_cover_args: tuple[DikeTrajectInfo, DikeSection, MeasureProtocol],
        expected_calculator_type: type[VerticalPipingMeasureCalculatorProtocol],
    ):
        # 1. Define test data.
        _dike_traject, _dike_section, _measure = calculator_d_cover_args

        # 2. Run test.
        _calculator = VerticalPipingMeasureCalculatorFactory.get_calculator(
            _dike_traject, _dike_section, _measure
        )

        # 3. Verify expectations.
        assert isinstance(_calculator, expected_calculator_type)
        assert isinstance(_calculator, VerticalPipingMeasureCalculatorProtocol)

    @pytest.mark.parametrize(
        "calculator_d_cover_args",
        [
            pytest.param(None, id="NONE provided"),
            pytest.param(float("nan"), id="NaN provided"),
        ],
        indirect=["calculator_d_cover_args"],
    )
    def test_given_invalid_d_cover_value_raises(
        self,
        calculator_d_cover_args: tuple[DikeTrajectInfo, DikeSection, MeasureProtocol],
    ):
        # 1. Define test data.
        _dike_traject, _dike_section, _measure = calculator_d_cover_args

        # 2. Run test.
        with pytest.raises(TypeError) as exc_err:
            VerticalPipingMeasureCalculatorFactory.get_calculator(
                _dike_traject, _dike_section, _measure
            )

        # 3. Verify expectations.
        assert (
            str(exc_err.value)
            == f"Not supported `d_cover` value ({_dike_section.cover_layer_thickness})"
        )
