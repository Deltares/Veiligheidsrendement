import dataclasses
from math import isnan
from typing import Iterator

import pytest
from pandas import concat, read_csv

from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults import default_unit_costs_csv


def get_required_unformatted_field_names() -> list[str]:
    """
    Define expected keys based on fields without a default value.
    """
    return [
        _field.name.capitalize().replace("_", " ")
        for _field in dataclasses.fields(MeasureUnitCosts)
        if (_field.default == dataclasses.MISSING)
        and (_field.default_factory == dataclasses.MISSING)
    ]


def get_unformatted_unit_cost_dict() -> dict:
    _unit_cost_data = read_csv(str(default_unit_costs_csv), encoding="latin_1")
    _unit_cost_dict = {}
    for _, _series in _unit_cost_data.iterrows():
        _unit_cost_dict[_series["Description"]] = _series["Cost"]
    return _unit_cost_dict


class TestMeasureUnitCosts:
    @pytest.fixture(name="required_unformatted_field_names")
    def _get_required_unformatted_field_names_fixture(self) -> Iterator[list[str]]:
        yield get_required_unformatted_field_names()

    @pytest.fixture(name="valid_unformatted_dict")
    def _get_valid_unformatted_dict_fixture(
        self, required_unformatted_field_names: list[str]
    ) -> Iterator[dict]:
        yield {k: 4.2 for k in required_unformatted_field_names}

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

    def test_initialize_with_extra_keys_raises_error(
        self, valid_unformatted_dict: dict
    ):
        # test that adding extra keys that are not recognized raises an error
        # 1. Define test data.
        _extra_key = "Dummy Key"
        assert _extra_key not in valid_unformatted_dict

        valid_unformatted_dict[_extra_key] = 4.2

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 3. Verify expectations.
        _expected_error = "Kosten voor maatregel '{}' gevonden, maar niet herkend in de VRTOOL. Controleer de waarden en pas deze aan in het bestand unit_costs.csv.".format(
            _extra_key
        )

        assert str(exc_err.value) == _expected_error

    def test_given_decreasing_cost_for_block_when_from_unformatted_dict_then_raises(
        self, valid_unformatted_dict: dict
    ):
        # 1. Define test data.
        # Add two costs for Installation of blocks where costs for larger blocks are lower
        valid_unformatted_dict["Installation of blocks 10cm"] = 4.2
        valid_unformatted_dict["Installation of blocks 20cm"] = 3.2

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)

        # 3. Verify expectations.
        _expected_error = "Kosten voor installatie blokken dalen met toenemende dikte. Controleer de waarden en pas deze aan in het bestand unit_costs.csv."
        assert str(exc_err.value) == _expected_error

    def test_given_unsorted_cost_for_block_when_from_unformatted_dict_then_initializes(
        self, valid_unformatted_dict: dict
    ):
        # 1. Define test data.
        # add two costs for Installation of blocks where block thicknesses are not sorted but costs increase
        valid_unformatted_dict["Installation of blocks 20cm"] = 4.2
        valid_unformatted_dict["Installation of blocks 10cm"] = 3.2

        # 2. Run test.
        _unit_costs = MeasureUnitCosts.from_unformatted_dict(valid_unformatted_dict)
        assert isinstance(_unit_costs, MeasureUnitCosts)

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

    def test_init_from_modified_csv_file_bad_key(self):
        # 1. Define test data using part of classmethod
        _unformatted_dict = get_unformatted_unit_cost_dict()

        # 2. Modify the data: mess up key
        # modify keys that contain 'Installation of blocks' to 'Installatie blokken'
        _keys_to_change = list(
            filter(lambda x: "Installation of blocks" in x, _unformatted_dict.keys())
        )
        # change the keys
        for _key in _keys_to_change:
            _unformatted_dict[
                _key.replace("Installation of blocks", "Installatie blokken")
            ] = _unformatted_dict.pop(_key)

        _first_bad_key = _keys_to_change[0].replace(
            "Installation of blocks", "Installatie blokken"
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            MeasureUnitCosts.from_unformatted_dict(_unformatted_dict)

        # 3. Verify expectations.
        _expected_error = "Kosten voor maatregel '{}' gevonden, maar niet herkend in de VRTOOL. Controleer de waarden en pas deze aan in het bestand unit_costs.csv.".format(
            _first_bad_key
        )
        assert str(exc_err.value) == _expected_error

    def test_identical_thickness_for_block_crashes(self):
        # 1. Define test data using read_csv
        _cost_data = read_csv(str(default_unit_costs_csv), encoding="latin_1")
        # duplicate a line of costs for Installation of blocks
        _block_key = "Installation of blocks (D=30cm)"
        _cost_data = concat(
            [_cost_data, _cost_data.loc[_cost_data["Description"] == _block_key]],
            ignore_index=True,
        )

        # 2. Run test.
        # verify it crashes
        with pytest.raises(ValueError) as exc_err:
            MeasureUnitCosts.cost_dataframe_to_dict(_cost_data)

        _expected_error = "Dubbele kosten gevonden voor {} in unit_costs.csv. Controleer de waarden en pas deze aan.".format(
            _block_key
        )

        assert str(exc_err.value) == _expected_error
