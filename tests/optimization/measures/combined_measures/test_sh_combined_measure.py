from typing import Callable, Iterator

import pytest

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestShCombinedMeasure:
    def test_initialize(self, sh_measure_factory: Callable[[], ShMeasure]):
        # 1. Define test data.
        _sh_measure = sh_measure_factory()
        assert isinstance(_sh_measure, ShMeasure)

        # 2. Run test.
        _combined_measure = ShCombinedMeasure(
            primary=_sh_measure, mechanism_year_collection=None
        )

        # 3. Verify expectations
        assert isinstance(_combined_measure, CombinedMeasureBase)
        assert isinstance(_combined_measure, ShCombinedMeasure)

    @pytest.fixture(name="sh_combined_measure_fixture")
    def _get_sh_combined_measure_fixture(
        self, sh_measure_factory: Callable[[], ShMeasure]
    ) -> Iterator[ShCombinedMeasure]:
        # Modify some of the parameters for more insightful tests.
        _sg_measure = sh_measure_factory()
        _sg_measure.base_cost = 100
        _sg_measure.cost = 420
        _sg_measure.year = 0

        yield ShCombinedMeasure(primary=_sg_measure, mechanism_year_collection=None)

    @pytest.mark.parametrize(
        "base_cost",
        [pytest.param(0, id="Cost is 0"), pytest.param(100, id="Cost is 100")],
    )
    def test_lcc_takes_base_cost(
        self, base_cost: float, sh_combined_measure_fixture: ShCombinedMeasure
    ):
        # 1. Define test data.
        sh_combined_measure_fixture.primary.base_cost = (
            base_cost + sh_combined_measure_fixture.primary.cost
        )
        _expected_lcc = (
            sh_combined_measure_fixture.primary.cost
            - sh_combined_measure_fixture.primary.base_cost
        )

        # 2. Run test.
        assert sh_combined_measure_fixture.lcc == _expected_lcc

    def test_lcc_varies_with_different_primary_year(
        self, sh_combined_measure_fixture: ShCombinedMeasure
    ):
        # 1. Define test data.
        _lcc_year_0 = sh_combined_measure_fixture.lcc

        # 2. Run test
        sh_combined_measure_fixture.primary.year = 20

        # 3. Verify expectations
        assert _lcc_year_0 != sh_combined_measure_fixture.lcc

    def test_lcc_with_secondary(
        self,
        sh_measure_factory: Callable[[], ShMeasure],
        sh_combined_measure_fixture: ShCombinedMeasure,
    ):
        # 1. Define test data.
        sh_combined_measure_fixture.primary.discount_rate = 0
        sh_combined_measure_fixture.primary.base_cost = 0
        _sh_measure = sh_measure_factory()
        _sh_measure.discount_rate = 0
        _sh_measure.year = 0
        _sh_measure.cost = 2 * sh_combined_measure_fixture.primary.cost

        # 2. Run test.
        sh_combined_measure_fixture.secondary = _sh_measure

        # 3. Verify expectations
        assert sh_combined_measure_fixture.lcc == pytest.approx(
            3 * sh_combined_measure_fixture.primary.cost, 0.001
        )
