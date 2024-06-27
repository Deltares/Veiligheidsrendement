from copy import deepcopy
from typing import Callable, Iterable, Iterator

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
from vrtool.optimization.measures.combined_measures.shsg_combined_measure import (
    ShSgCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


class TestCombinedMeasureFactory:
    @pytest.fixture(name="to_combine_measure_factory")
    def _get_secondary_measure_factory(
        self, measure_as_input_factory
    ) -> Iterable[Callable[[int], MeasureAsInputProtocol]]:
        def create_measure_as_input(measure_result_id: int) -> MeasureAsInputProtocol:
            return measure_as_input_factory(
                **dict(
                    measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                    measure_result_id=measure_result_id,
                )
            )

        yield create_measure_as_input

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
        self,
        to_combine_measure_factory: Callable[[int], MeasureAsInputProtocol],
        request: pytest.FixtureRequest,
    ) -> tuple[
        MeasureAsInputProtocol, MeasureAsInputProtocol, type[CombinedMeasureBase]
    ]:
        _measure_type, _measure_type_dict, _expected_combined_type = request.param
        _primary_measure_result_id = 2
        _secondary_measure_result_id = 3
        _secondary = to_combine_measure_factory(_secondary_measure_result_id)
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

    def test_from_input_with_unsupported_type_raises(
        self, to_combine_measure_factory: Callable[[int], MeasureAsInputProtocol]
    ):
        # 1. Define test data.
        _base_measure = to_combine_measure_factory(23)
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

    @pytest.fixture(name="shsg_combined_measure_arguments")
    def _get_shsg_combined_measure_arguments_fixture(
        self,
        section_with_combinations: SectionAsInput,
        shsg_measure_factory: Callable[[], ShSgMeasure],
        to_combine_measure_factory: Callable[[int], MeasureAsInputProtocol],
    ) -> Iterator[tuple[ShSgMeasure, ShCombinedMeasure, SgCombinedMeasure]]:

        # Define the `ShCombinedMeasure`
        assert any(section_with_combinations.sh_combinations)
        _sh_combined = section_with_combinations.sh_combinations[0]
        _sh_combined.secondary = to_combine_measure_factory(23)
        assert isinstance(_sh_combined.secondary, MeasureAsInputProtocol)

        # Define the `SgCombinedMeasure`
        assert any(section_with_combinations.sg_combinations)
        _sg_combined = section_with_combinations.sg_combinations[0]
        _sg_combined.secondary = to_combine_measure_factory(12)
        assert isinstance(_sg_combined.secondary, MeasureAsInputProtocol)

        # Define the `ShSgmeasure`
        _shsg_measure = shsg_measure_factory()
        _shsg_measure.dcrest = _sh_combined.primary.dcrest
        _shsg_measure.dberm = _sg_combined.primary.dberm
        _shsg_measure.l_stab_screen = _sg_combined.primary.l_stab_screen
        _shsg_measure.measure_type = _sh_combined.primary.measure_type
        _shsg_measure.year = _sh_combined.primary.year

        yield (_shsg_measure, _sh_combined, _sg_combined)

    def test_get_shsg_combined_measure_with_different_years_returns_none(
        self,
        shsg_combined_measure_arguments: tuple[
            ShSgMeasure, ShCombinedMeasure, SgCombinedMeasure
        ],
    ):
        # 1. Define test data.
        (
            _shsg_measure,
            _sh_combined,
            _sg_combined,
        ) = shsg_combined_measure_arguments
        _shsg_measure.year = _sh_combined.primary.year + 100

        # 2. Run test.
        _shsg_measure = CombinedMeasureFactory.get_shsg_combined_measure(
            [_shsg_measure], _sh_combined, _sg_combined
        )

        # 3. Verify expectations.
        assert _shsg_measure is None

    def test_get_shsg_combined_measure_returns_shsgcombinedmeasure(
        self,
        shsg_combined_measure_arguments: tuple[
            ShSgMeasure, ShCombinedMeasure, SgCombinedMeasure
        ],
    ):
        # 1. Define test data.
        (
            _shsg_measure,
            _sh_combined,
            _sg_combined,
        ) = shsg_combined_measure_arguments

        # 2. Run test.
        _shsg_combined_measure = CombinedMeasureFactory.get_shsg_combined_measure(
            [_shsg_measure], _sh_combined, _sg_combined
        )

        # 3. Verify expectations.
        assert isinstance(_shsg_combined_measure, ShSgCombinedMeasure)
        assert _shsg_combined_measure.primary == _shsg_measure
        assert _shsg_combined_measure.sh_secondary == _sh_combined.secondary
        assert _shsg_combined_measure.sg_secondary == _sg_combined.secondary
        assert (
            _shsg_combined_measure.mechanism_year_collection
            == _shsg_measure.mechanism_year_collection
        )

    def test_get_shsg_combined_measure_multiple_options_returns_first_found(
        self,
        shsg_combined_measure_arguments: tuple[
            ShSgMeasure, ShCombinedMeasure, SgCombinedMeasure
        ],
    ):
        # 1. Define test data.
        (
            _shsg_measure,
            _sh_combined,
            _sg_combined,
        ) = shsg_combined_measure_arguments
        _alter_shsg_measure = deepcopy(_shsg_measure)
        assert _shsg_measure != _alter_shsg_measure

        # 2. Run test.
        _shsg_combined_measure = CombinedMeasureFactory.get_shsg_combined_measure(
            [_alter_shsg_measure, _shsg_measure], _sh_combined, _sg_combined
        )

        # 3. Verify expectations.
        assert isinstance(_shsg_combined_measure, ShSgCombinedMeasure)
        assert _shsg_combined_measure.primary == _alter_shsg_measure
        assert _shsg_combined_measure.sh_secondary == _sh_combined.secondary
        assert _shsg_combined_measure.sg_secondary == _sg_combined.secondary
        assert (
            _shsg_combined_measure.mechanism_year_collection
            == _alter_shsg_measure.mechanism_year_collection
        )
