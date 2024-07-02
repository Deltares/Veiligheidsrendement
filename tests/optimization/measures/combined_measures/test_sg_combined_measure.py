from typing import Callable, Iterator

import pytest

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.sg_measure import SgMeasure


class TestSgCombinedMeasure:
    def test_initialize(self, sg_measure_factory: Callable[[], SgMeasure]):
        # 1. Define test data.
        _sg_measure = sg_measure_factory()
        assert isinstance(_sg_measure, SgMeasure)

        # 2. Run test.
        _combined_measure = SgCombinedMeasure(
            primary=_sg_measure, mechanism_year_collection=None
        )

        # 3. Verify expectations
        assert isinstance(_combined_measure, CombinedMeasureBase)
        assert isinstance(_combined_measure, SgCombinedMeasure)

    @pytest.fixture(name="sg_combined_measure_fixture")
    def _get_sg_combined_measure_fixture(
        self, sg_measure_factory: Callable[[], SgMeasure]
    ) -> Iterator[SgCombinedMeasure]:
        # Modify some of the parameters for more insightful tests.
        _sg_measure = sg_measure_factory()
        _sg_measure.base_cost = 100
        _sg_measure.cost = 420
        _sg_measure.year = 0

        yield SgCombinedMeasure(primary=_sg_measure, mechanism_year_collection=None)

    @pytest.mark.parametrize(
        "base_cost",
        [pytest.param(0, id="Cost is 0"), pytest.param(100, id="Cost is 100")],
    )
    def test_lcc_does_not_take_base_cost(
        self, base_cost: float, sg_combined_measure_fixture: SgCombinedMeasure
    ):
        # 1. Define test data.
        sg_combined_measure_fixture.primary.base_cost = (
            base_cost + sg_combined_measure_fixture.primary.cost
        )

        # 2. Run test.
        assert (
            sg_combined_measure_fixture.lcc == sg_combined_measure_fixture.primary.cost
        )

    def test_lcc_varies_with_different_primary_year(
        self, sg_combined_measure_fixture: SgCombinedMeasure
    ):
        # 1. Define test data.
        _lcc_year_0 = sg_combined_measure_fixture.lcc

        # 2. Run test
        sg_combined_measure_fixture.primary.year = 20

        # 3. Verify expectations
        assert _lcc_year_0 != sg_combined_measure_fixture.lcc

    def test_lcc_with_secondary(
        self,
        sg_measure_factory: Callable[[], SgMeasure],
        sg_combined_measure_fixture: SgCombinedMeasure,
    ):
        # 1. Define test data.
        sg_combined_measure_fixture.primary.discount_rate = 0
        _sg_measure = sg_measure_factory()
        _sg_measure.discount_rate = 0
        _sg_measure.year = 0
        _sg_measure.cost = 2 * sg_combined_measure_fixture.primary.cost

        # 2. Run test.
        sg_combined_measure_fixture.secondary = _sg_measure

        # 3. Verify expectations
        assert sg_combined_measure_fixture.lcc == pytest.approx(
            3 * sg_combined_measure_fixture.primary.cost, 0.001
        )
