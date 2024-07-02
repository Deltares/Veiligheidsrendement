from dataclasses import dataclass

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.sg_measure import SgMeasure


@dataclass(kw_only=True)
class SgCombinedMeasure(CombinedMeasureBase):
    """
    Represents the combination between two supporting `SgMeasure`
    where the most important (`primary`) will have the leading
    base costs over the auxiliar (`secondary`) one.
    """

    primary: SgMeasure
    secondary: SgMeasure | None = None

    @property
    def _base_cost(self) -> float:
        return 0

    def _get_secondary_lcc(self) -> float:
        if self.secondary is None:
            return 0
        # secondary lcc is calculated with base_cost = 0
        return self._calculate_lcc(self.secondary, 0)
