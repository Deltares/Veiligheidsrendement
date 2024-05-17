import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure


class TestSgMeasure:
    def _create_sg_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> SgMeasure:
        _measure = SgMeasure(
            measure_result_id=42,
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=10.5,
            year=10,
            discount_rate=0.03,
            mechanism_year_collection=None,
            dberm=0.1,
            l_stab_screen=float("nan"),
        )
        return _measure

    def test_create_sg_measure(self):
        # 1. Define input
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE

        # 2. Run test
        _measure = self._create_sg_measure(_measure_type, _combine_type)

        # 3. Verify expectations
        assert isinstance(_measure, SgMeasure)
        assert isinstance(_measure, MeasureAsInputProtocol)
        assert _measure.measure_type == _measure_type
        assert _measure.combine_type == _combine_type
        assert _measure.cost == pytest.approx(10.5)
        assert _measure.year == 10
        assert _measure.discount_rate == pytest.approx(0.03)
        assert _measure.mechanism_year_collection is None
        assert _measure.dberm == pytest.approx(0.1)
        assert _measure.start_cost == pytest.approx(0)

    @pytest.mark.parametrize("dberm_value", [pytest.param(0), pytest.param(-999)])
    def test_given_dberm_0_lcc_returns_0(self, dberm_value: float):
        """
        Test related to issue VRTOOL-390
        """
        # 1. Define test data.
        # Measure and combinable type do not really matter,
        # but we are forced to set a value.
        _sg_measure = self._create_sg_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
        )
        _sg_measure.dberm = dberm_value

        # 2. Run test.
        _result = _sg_measure.lcc

        # 3. Verify final expectations.
        assert _result == 0

    @pytest.mark.parametrize(
        "dberm_value",
        [pytest.param(-10, id="Smaller than 0"), pytest.param(10, id="Greater than 0")],
    )
    def test_given_dberm_else_than_0_lcc_doesnot_return_0(self, dberm_value: float):
        """
        Test related to issue VRTOOL-390
        """
        # 1. Define test data.
        # Measure and combinable type do not really matter,
        # but we are forced to set a value.
        _sg_measure = self._create_sg_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
        )
        _sg_measure.dberm = dberm_value

        # 2. Run test.
        _result = _sg_measure.lcc

        # 3. Verify final expectations.
        assert _result != 0

    def test_given_custom_measure_without_dberm_returns_cost(self):
        """
        Test related to issue VRTOOL-510
        """
        # 1. Define test data.
        # Measure and combinable type do not really matter,
        # but we are forced to set a value.
        _sg_measure = self._create_sg_measure(
            MeasureTypeEnum.CUSTOM, CombinableTypeEnum.COMBINABLE
        )
        _sg_measure.dberm = 0

        # 2. Run test.
        _result = _sg_measure.lcc

        # 3. Verify final expectations.
        assert _result > 0

    @pytest.mark.parametrize(
        "measure_type",
        [
            pytest.param(mt)
            for mt in MeasureTypeEnum
            if mt
            not in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
            ]
        ],
    )
    def test_given_unsupported_measure_type_when_set_start_cost_value_is_set_to_0(
        self, measure_type: MeasureTypeEnum
    ):
        # 1. Define test data.
        _combine_type = CombinableTypeEnum.FULL
        _measure = self._create_sg_measure(measure_type, _combine_type)
        assert _measure.start_cost == pytest.approx(0)

        # 2. Run test.
        _measure.start_cost = 42

        # 3. Verify expectations.
        assert _measure.start_cost == pytest.approx(0)

    @pytest.mark.parametrize(
        "measure_type",
        [
            pytest.param(MeasureTypeEnum.SOIL_REINFORCEMENT),
            pytest.param(MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN),
        ],
    )
    def test_given_supported_measure_type_when_set_start_cost_value_is_set_to_0(
        self, measure_type: MeasureTypeEnum
    ):
        # 1. Define test data.
        _combine_type = CombinableTypeEnum.FULL
        _measure = self._create_sg_measure(measure_type, _combine_type)
        assert _measure.start_cost == pytest.approx(0)
        _new_value = 42

        # 2. Run test.
        _measure.start_cost = _new_value

        # 3. Verify expectations.
        assert _measure.start_cost == pytest.approx(_new_value)

    def test_lcc(self):
        # 1. Define input
        _measure = self._create_sg_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, CombinableTypeEnum.COMBINABLE
        )
        _measure.start_cost = 5.5

        # 2. Run test
        _lcc = _measure.lcc

        # 3. Verify expectations
        assert _lcc == pytest.approx(3.720469)

    @pytest.mark.parametrize(
        "mechanism, expected",
        [
            pytest.param(MechanismEnum.PIPING, True, id="VALID PIPING"),
            pytest.param(MechanismEnum.OVERFLOW, False, id="INVALID OVERFLOW"),
        ],
    )
    def test_is_mechanism_allowed(self, mechanism: MechanismEnum, expected: bool):
        # 1./2. Define input & Run test
        _results = SgMeasure.is_mechanism_allowed(mechanism)

        # 3. Verify expectations
        assert _results == expected

    def test_get_allowed_mechanism(self):
        # 1./2. Define input & Run test
        _expected_mechanisms = [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]
        _allowed_mechanisms = SgMeasure.get_allowed_mechanisms()

        # 3. Verify expectations
        assert len(_expected_mechanisms) == len(_allowed_mechanisms)
        assert all(x in _expected_mechanisms for x in _allowed_mechanisms)

    def test_get_allowed_measure_combination(self):
        # 1./2. Define input & Run test
        _allowed_combinations = SgMeasure.get_allowed_measure_combinations()

        # 3. Verify expectations
        assert isinstance(_allowed_combinations, dict)
        assert _allowed_combinations

    @pytest.mark.parametrize(
        "dberm_value",
        [
            pytest.param(0),
            pytest.param(float("nan")),
        ],
    )
    def test_is_initial_cost_measure_when_year_0_and_dberm_0_or_nan_then_is_true(
        self, dberm_value: float
    ):
        # 1. Define test data.
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE
        _measure = self._create_sg_measure(_measure_type, _combine_type)
        _measure.dberm = dberm_value
        _measure.year = 0

        # 2. Run test and verify expectations.
        assert _measure.is_initial_cost_measure() is True

    @pytest.mark.parametrize(
        "dberm_value",
        [
            pytest.param(0),
            pytest.param(float("nan")),
        ],
    )
    def test_is_initial_cost_measure_when_year_not_0_and_dberm_0_or_nan_then_is_false(
        self, dberm_value: float
    ):
        # 1. Define test data.
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE
        _measure = self._create_sg_measure(_measure_type, _combine_type)
        _measure.year = 4

        # 2. Run test and verify expectations.
        assert _measure.is_initial_cost_measure() is False

    def test_is_initial_cost_measure_when_year_0_and_dberm_not_0_or_not_nan_then_is_false(
        self,
    ):
        # 1. Define test data.
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE
        _measure = self._create_sg_measure(_measure_type, _combine_type)
        _measure.year = 0

        # 2. Run test and verify expectations.
        assert _measure.is_initial_cost_measure() is False
