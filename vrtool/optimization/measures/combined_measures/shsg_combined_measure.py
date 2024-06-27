from dataclasses import dataclass

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


@dataclass(kw_only=True)
class ShSgCombinedMeasure(CombinedMeasureBase):
    """
    This `CombinedMeasure` specialization brings together a `ShSgMeasure`
    with the secondary measures of its matching `SgMeasure` and `ShMeasure`.
    """

    primary: ShSgMeasure
    sh_secondary: ShMeasure
    sg_secondary: SgMeasure

    @property
    def _base_cost(self) -> float:
        return 0

    def is_base_measure(self) -> bool:
        return False

    def _get_secondary_lcc(self) -> float:
        _secondary_costs = 0
        if self.sh_secondary:
            _secondary_costs += self._calculate_lcc(self.sh_secondary, 0)

        if self.sg_secondary:
            _secondary_costs += self._calculate_lcc(self.sg_secondary, 0)

        return _secondary_costs
