import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.combined_measure_factory import (
    CombinedMeasureFactory,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestCombinedMeasureFactory:
    @pytest.fixture(
        name="from_input_cases",
        params=[
            [
                ShMeasure,
                dict(
                    beta_target=float("nan"),
                    transition_level=float("nan"),
                    dcrest=float("nan"),
                ),
                ShCombinedMeasure,
            ],
            [SgMeasure, dict(dberm=float("nan")), SgCombinedMeasure],
        ],
        ids=[
            "Primary ShMeasre -> ShCombinedMeasure",
            "Primary SgMeasure -> SgCombinedMeasure",
        ],
    )
    def _get_from_input_cases_fixture(
        self, measure_as_input_factory, request: pytest.FixtureRequest
    ) -> tuple[
        MeasureAsInputProtocol, MeasureAsInputProtocol, type[CombinedMeasureBase]
    ]:
        _measure_type, _measure_type_dict, _expected_combined_type = request.param
        _primary_measure_result_id = 2
        _secondary_measure_result_id = 3
        _secondary = measure_as_input_factory(
            **dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=_secondary_measure_result_id,
            )
        )
        _primary = _measure_type(
            **(
                _measure_type_dict
                | _secondary.__dict__
                | dict(measure_result_id=_primary_measure_result_id)
            ),
        )
        return (_primary, _secondary, _expected_combined_type)

    def test_from_input_with_secondary_measure(
        self,
        from_input_cases: tuple[MeasureAsInputProtocol, MeasureAsInputProtocol],
    ):
        # 1. Define test data
        _primary, _secondary, _expected_combined_type = from_input_cases
        assert isinstance(_primary, MeasureAsInputProtocol)
        assert isinstance(_secondary, MeasureAsInputProtocol)
        _sequence_nr = 7

        # 2. Run test
        _combination = CombinedMeasureFactory.from_input(
            _primary,
            _secondary,
            _primary.mechanism_year_collection,
            _sequence_nr,
        )

        # 3. Verify expectations
        assert isinstance(_combination, CombinedMeasureBase)
        assert isinstance(_combination, _expected_combined_type)
        assert _combination.primary.measure_result_id == _primary.measure_result_id
        assert _combination.secondary.measure_result_id == _secondary.measure_result_id
        assert _combination.sequence_nr == 7

    def test_from_input_without_secondary_measure(
        self,
        from_input_cases: tuple[MeasureAsInputProtocol, MeasureAsInputProtocol],
    ):
        # 1. Define test data
        _primary, _, _expected_combined_type = from_input_cases
        assert isinstance(_primary, MeasureAsInputProtocol)
        _sequence_nr = 7

        # 2. Run test
        _combination = CombinedMeasureFactory.from_input(
            _primary,
            None,
            _primary.mechanism_year_collection,
            _sequence_nr,
        )

        # 3. Verify expectations
        assert isinstance(_combination, CombinedMeasureBase)
        assert isinstance(_combination, _expected_combined_type)
        assert _combination.primary.measure_result_id == _primary.measure_result_id
        assert _combination.secondary == None
        assert _combination.sequence_nr == 7

    def test_from_input_with_unsupported_type_raises(self, measure_as_input_factory):
        # 1. Define test data.
        _base_measure = measure_as_input_factory(
            **dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=23,
            )
        )
        assert isinstance(_base_measure, MeasureAsInputBase)
        _expected_error = f"It is not supported to combine measures of type {MeasureAsInputBase.__name__}."

        # 2. Run test.
        with pytest.raises(NotImplementedError) as exc_err:
            CombinedMeasureFactory.from_input(
                primary=_base_measure,
                secondary=None,
                initial_assessment=None,
                sequence_nr=42,
            )

        # 3. Verify expectations
        assert str(exc_err.value) == _expected_error
