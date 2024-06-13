import math
from typing import Callable

import pytest

from vrtool.orm.models import MeasureResult
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter,
)
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestMeasureResult:
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture(
        self, get_basic_measure_per_section: Callable[[], MeasurePerSection]
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()

        # 2. Run test
        _measure_result = MeasureResult.create(
            beta=3.1234,
            time=0.0,
            cost=100,
            measure_per_section=_measure_per_section,
        )

        # 3. Verify expectations.
        assert isinstance(_measure_result, MeasureResult)
        assert isinstance(_measure_result, OrmBaseModel)
        assert _measure_result.measure_per_section == _measure_per_section
        assert _measure_result in _measure_per_section.measure_per_section_result

    @pytest.mark.usefixtures("empty_db_fixture")
    def test_given_measure_result_without_parameters_nan_is_returned(
        self, get_basic_measure_per_section: Callable[[], MeasurePerSection]
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()
        _measure_result = MeasureResult.create(
            beta=3.1234,
            time=0,
            cost=100.54,
            measure_per_section=_measure_per_section,
        )

        # 2. Run test.
        _result_value = _measure_result.get_parameter_value("NotAParmeter")

        # 3. Verify expectations.
        assert math.isnan(_result_value)

    @pytest.mark.parametrize(
        "string_variation",
        [
            pytest.param(str.capitalize, id="Capitalize name"),
            pytest.param(str.lower, id="Lowercase name"),
            pytest.param(str.upper, id="Uppercase name"),
        ],
    )
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_given_measure_result_with_parameters_returns_value(
        self,
        string_variation: Callable,
        get_basic_measure_per_section: Callable[[], MeasurePerSection],
    ):
        # 1. Define test data.
        _measure_per_section = get_basic_measure_per_section()
        _measure_result = MeasureResult.create(
            beta=3.1234,
            time=0.0,
            cost=100,
            measure_per_section=_measure_per_section,
        )
        _parameter_value = 4.2
        _parameter_name = "dummyparameter"
        MeasureResultParameter.create(
            name=string_variation(_parameter_name),
            value=4.2,
            measure_result=_measure_result,
        )

        # 2. Run test.
        _result_value = _measure_result.get_parameter_value(_parameter_name)

        # 3. Verify expectations.
        assert _result_value == pytest.approx(_parameter_value)
