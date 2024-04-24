from math import isnan

import pytest

from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults import default_unit_costs_csv

_expected_keys = [
    "Inward starting costs",
    "Inward added volume",
    "Outward added volume",
    "Outward reused volume",
    "Outward reuse factor",
    "Outward compensation factor",
    "Outward removed volume",
    "Road renewal",
    "House removal",
]


class TestMeasureUnitCosts:
    @pytest.fixture
    def valid_unformatted_dict(self) -> dict:
        return {k: 4.2 for k in _expected_keys}

    def test_initialize_from_unformatted_dict(self, valid_unformatted_dict: dict):
        # 1. Define test data.
        assert len(valid_unformatted_dict.keys()) == len(_expected_keys)
        assert set(valid_unformatted_dict.keys()) == set(_expected_keys)

        # 2. Run test.
        _unit_costs = MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 3. Verify final expectations.
        assert isinstance(_unit_costs, MeasureUnitCosts)

        # Inward costs
        assert _unit_costs.inward_added_volume == 4.2
        assert _unit_costs.inward_starting_costs == 4.2

        # Outward costs
        assert _unit_costs.outward_reuse_factor == 4.2
        assert _unit_costs.outward_removed_volume == 4.2
        assert _unit_costs.outward_reused_volume == 4.2
        assert _unit_costs.outward_added_volume == 4.2
        assert _unit_costs.outward_compensation_factor == 4.2

        # Other
        assert _unit_costs.house_removal == 4.2
        assert _unit_costs.road_renewal == 4.2

        # Optionals
        assert isnan(_unit_costs.sheetpile)
        assert isnan(_unit_costs.diaphragm_wall)
        assert isnan(_unit_costs.vertical_geotextile)

    def test_initialize_with_extra_keys_raises_error(
        self, valid_unformatted_dict: dict
    ):
        # 1. Define test data.
        assert len(valid_unformatted_dict.keys()) == len(_expected_keys)
        assert set(valid_unformatted_dict.keys()) == set(_expected_keys)

        _extra_key = "Dummy Key"
        assert _extra_key not in valid_unformatted_dict

        valid_unformatted_dict[_extra_key] = 4.2

        # 2. Run test.
        with pytest.raises(TypeError) as exc_err:
            MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 3. Verify expectations.
        _expected_error = (
            "MeasureUnitCosts.__init__() got an unexpected keyword argument 'dummy_key'"
        )
        assert str(exc_err.value) == _expected_error

    @pytest.mark.parametrize(
        "excluded_key",
        [pytest.param(_key, id="Without '{}'".format(_key)) for _key in _expected_keys],
    )
    def test_initialize_without_required_key_raises_error(self, excluded_key: str):
        # 1. Define test data.
        assert excluded_key in _expected_keys
        _unformatted_dict = {k: 4.2 for k in _expected_keys if k != excluded_key}

        # 2. Run test.
        with pytest.raises(TypeError) as exc_err:
            MeasureUnitCosts.from_unformatted_dict(_unformatted_dict)

        # 3. Verify expectations.
        _expected_error = (
            "MeasureUnitCosts.__init__() missing 1 required positional argument: '{}'"
        )
        _formatted_key = excluded_key.strip().lower().replace(" ", "_")
        assert str(exc_err.value) == _expected_error.format(_formatted_key)

    def test_initialize_from_default_csv_file(self):
        # 1. Define test data.
        assert default_unit_costs_csv.is_file()

        # 2. Run test.
        _unit_costs = MeasureUnitCosts.from_csv_file(default_unit_costs_csv)

        # 3. Verify expectations
        assert isinstance(_unit_costs, MeasureUnitCosts)
