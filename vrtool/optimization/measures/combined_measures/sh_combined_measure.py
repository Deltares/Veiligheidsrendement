from dataclasses import dataclass

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass(kw_only=True)
class ShCombinedMeasure(CombinedMeasureBase):
    """
    Represents the combination between two supporting `ShMeasure`
    where the most important (`primary`) will have the leading
    base costs over the auxiliar (`secondary`) one.
    """

    primary: ShMeasure
    secondary: ShMeasure | None = None

    @property
    def _base_cost(self) -> float:
        return self.primary.base_cost

    def get_secondary_lcc(self) -> float:
        if self.secondary is None:
            return 0
        # secondary lcc is calculated with base_cost = 0
        return self._calculate_lcc(self.secondary, 0)
