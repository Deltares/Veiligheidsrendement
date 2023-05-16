import math

from vrtool.orm.io.importers.converters import to_valid_float


class TestConverters:
    def test_none_to_valid_float(self):
        _return_value = to_valid_float(None)
        assert math.isnan(_return_value)

    def test_float_to_valid_float(self):
        _float_value = 4.2
        _return_value = to_valid_float(_float_value)
        assert _return_value == _float_value
