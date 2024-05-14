from vrtool.orm.io.exporters.measures.custom_measure_time_beta_calculator import (
    CustomMeasureTimeBetaCalculator,
)


class TestGetInterpolatedTimeBetaCollection:
    def test_given_one_custom_value_becomes_constant(self):
        # 1. Define test data.
        _constant_value = 4.2
        _custom_values = [(0, _constant_value)]
        _computation_periods = list(range(0, 10, 2))

        # 2. Run test.
        _interpolated_values = (
            CustomMeasureTimeBetaCalculator.get_interpolated_time_beta_collection(
                _custom_values, _computation_periods
            )
        )

        # 3. Verify expectations.
        assert len(_interpolated_values) == len(_computation_periods)
        assert all(
            _value == _constant_value for _value in _interpolated_values.values()
        )

    def test_given_multiple_values_last_one_becomes_constant(self):
        # 1. Define test data.
        _constant_value = 2
        _max_beta = 8
        _max_time = 6
        _step = 2
        _custom_values = [(0, _max_beta), (_max_time, _constant_value)]
        _computation_periods = list(range(0, _max_beta + (3 * _step), _step))

        _expected_values = dict()
        for _idx, _t in enumerate(_computation_periods):
            _expected_values[_t] = _max_beta - (_idx * _step)
            if _t > _max_time:
                _expected_values[_t] = _constant_value

        # 2. Run test.
        _interpolated_values = (
            CustomMeasureTimeBetaCalculator.get_interpolated_time_beta_collection(
                _custom_values, _computation_periods
            )
        )

        # 3. Verify expectations.
        assert len(_interpolated_values) == len(_computation_periods)
        for _time, _beta in _interpolated_values.items():
            assert _beta == _expected_values[_time]
