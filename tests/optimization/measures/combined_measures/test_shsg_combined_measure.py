from typing import Callable, Iterator

import pytest

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.shsg_combined_measure import (
    ShSgCombinedMeasure,
)
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


class TestShSgCombinedMeasure:
    def test_initialize(self, shsg_measure_factory: Callable[[], ShSgMeasure]):
        # 1. Define test data.
        _sh_measure = shsg_measure_factory()
        assert isinstance(_sh_measure, ShSgMeasure)

        # 2. Run test.
        _combined_measure = ShSgCombinedMeasure(
            primary=_sh_measure,
            sh_secondary=None,
            sg_secondary=None,
            mechanism_year_collection=None,
        )

        # 3. Verify expectations
        assert isinstance(_combined_measure, CombinedMeasureBase)
        assert isinstance(_combined_measure, ShSgCombinedMeasure)
        assert _combined_measure.is_base_measure() is False

    @pytest.fixture(name="shsg_combined_measure_fixture")
    def _get_shsg_combined_measure_fixture(
        self, shsg_measure_factory: Callable[[], ShSgMeasure]
    ) -> Iterator[ShSgCombinedMeasure]:
        # Modify some of the parameters for more insightful tests.
        _shsg_measure = shsg_measure_factory()
        _shsg_measure.base_cost = 100
        _shsg_measure.cost = 420
        _shsg_measure.year = 0

        yield ShSgCombinedMeasure(
            primary=_shsg_measure,
            sh_secondary=None,
            sg_secondary=None,
            mechanism_year_collection=None,
        )

    @pytest.mark.parametrize(
        "base_cost",
        [pytest.param(0, id="Cost is 0"), pytest.param(100, id="Cost is 100")],
    )
    def test_lcc_does_not_take_base_cost(
        self, base_cost: float, shsg_combined_measure_fixture: ShSgCombinedMeasure
    ):
        # 1. Define test data.
        shsg_combined_measure_fixture.primary.base_cost = (
            base_cost + shsg_combined_measure_fixture.primary.cost
        )
        # 2. Run test.
        assert (
            shsg_combined_measure_fixture.lcc
            == shsg_combined_measure_fixture.primary.cost
        )

    def test_lcc_varies_with_different_primary_year(
        self, shsg_combined_measure_fixture: ShSgCombinedMeasure
    ):
        # 1. Define test data.
        # If there's no discount rate then the year does not really have an effect
        # as the formula is (1 + discount_rate)^year
        shsg_combined_measure_fixture.primary.discount_rate = 0.42
        _lcc_year_0 = shsg_combined_measure_fixture.lcc

        # 2. Run test
        shsg_combined_measure_fixture.primary.year = 20

        # 3. Verify expectations
        assert _lcc_year_0 != shsg_combined_measure_fixture.lcc

    @pytest.mark.parametrize(
        "with_sh_secondary",
        [
            pytest.param(True, id="With SH secondary"),
            pytest.param(False, id="Without SH secondary"),
        ],
    )
    @pytest.mark.parametrize(
        "with_sg_secondary",
        [
            pytest.param(True, id="With SG secondary"),
            pytest.param(False, id="Without SG secondary"),
        ],
    )
    def test_lcc_with_secondary(
        self,
        with_sh_secondary: bool,
        with_sg_secondary: bool,
        sh_measure_factory: Callable[[], ShSgMeasure],
        shsg_combined_measure_fixture: ShSgCombinedMeasure,
    ):
        # 1. Define test data.
        shsg_combined_measure_fixture.primary.discount_rate = 0
        shsg_combined_measure_fixture.primary.base_cost = 0
        _sh_measure = sh_measure_factory()
        _sh_measure.discount_rate = 0
        _sh_measure.year = 0
        _sh_measure.cost = 2 * shsg_combined_measure_fixture.primary.cost

        # 2. Run test.
        # We reuse in this case the secondary measure as we only care for the
        # resulting lcc value (so we can say lcc == 5 * primary.cost)
        shsg_combined_measure_fixture.sg_secondary = (
            _sh_measure if with_sh_secondary else None
        )
        shsg_combined_measure_fixture.sh_secondary = (
            _sh_measure if with_sg_secondary else None
        )

        # 3. Verify expectations
        _cost_increased = 1 + (2 * with_sh_secondary) + (2 * with_sg_secondary)
        assert shsg_combined_measure_fixture.lcc == pytest.approx(
            _cost_increased * shsg_combined_measure_fixture.primary.cost, 0.001
        )
