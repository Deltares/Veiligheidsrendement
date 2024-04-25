import dataclasses
from math import isnan

import pytest

from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults import default_unit_costs_csv


def get_required_unformatted_field_names() -> list[str]:
    """
    Define expected keys based on fields without a default value.
    """
    return [
        _field.name.capitalize().replace("_", " ")
        for _field in dataclasses.fields(MeasureUnitCosts)
        if _field.default == dataclasses.MISSING
    ]


class TestMeasureUnitCosts:
    @pytest.fixture
    def required_unformatted_field_names(self) -> list[str]:
        return get_required_unformatted_field_names()

    @pytest.fixture
    def valid_unformatted_dict(
        self, required_unformatted_field_names: list[str]
    ) -> dict:
        return {k: 4.2 for k in required_unformatted_field_names}

    def test_initialize_from_unformatted_dict(self, valid_unformatted_dict: dict):
        # 1. Run test

        _unit_costs = MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 2. Verify final expectations.
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

    def test_initialize_with_extra_keys_only_logs_error(
        self, valid_unformatted_dict: dict
    ):
        # 1. Define test data.
        _extra_key = "Dummy Key"
        assert _extra_key not in valid_unformatted_dict

        valid_unformatted_dict[_extra_key] = 4.2

        # 2. Run test.
        _unit_costs = MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 3. Verify expectations.
        assert isinstance(_unit_costs, MeasureUnitCosts)
        assert _extra_key not in _unit_costs.__dict__.keys()

    @pytest.mark.parametrize(
        "excluded_key",
        [
            pytest.param(_field, id="Excluding {}".format(_field))
            for _field in get_required_unformatted_field_names()
        ],
    )
    def test_initialize_without_required_key_raises_error(
        self, excluded_key: str, required_unformatted_field_names: list[str]
    ):
        # 1. Define test data.
        assert excluded_key in required_unformatted_field_names
        _unformatted_dict = {
            k: 4.2 for k in required_unformatted_field_names if k != excluded_key
        }

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
