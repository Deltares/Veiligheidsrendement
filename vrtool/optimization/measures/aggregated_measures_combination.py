from dataclasses import dataclass

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class AggregatedMeasureCombination:
    sh_combination: CombinedMeasure
    sg_combination: CombinedMeasure
    measure_result_id: int
    year: int

    @property
    def lcc(self):
        if (
            self.sh_combination.primary.measure_type
            == MeasureTypeEnum.SOIL_REINFORCEMENT
            and self.sh_combination.is_initial_measure()
            and self.sg_combination.is_initial_measure()
        ):
            return 0
        return (
            self.sh_combination.lcc
            + self.sg_combination.lcc
            - self.sh_combination.primary.base_cost
        ) / (1 + self.sh_combination.primary.discount_rate) ** self.year

    def check_primary_measure_result_id_and_year(
        self, primary_sh: ShMeasure, primary_sg: SgMeasure
    ) -> bool:
        """
        This method checks if the primary measure result id and year of the aggregated measure combination match the primary measure result id and year of the primary measures.

        Args:
            primary_sh (ShMeasure): The primary Sh measure of the section
            primary_sg (SgMeasure): The primary Sg measure of the section

        Returns:
            bool: True if it matches, False otherwise.

        """
        if (
            self.sh_combination.primary.measure_result_id
            == primary_sh.measure_result_id
        ) and (
            self.sg_combination.primary.measure_result_id
            == primary_sg.measure_result_id
        ):
            return (self.sh_combination.primary.year == primary_sh.year) and (
                self.sg_combination.primary.year == primary_sg.year
            )
        return False

    def get_combination_idx(self) -> tuple[int, int]:
        """
        Find the index of the Sh and Sg combination that compose the aggregate.

        Returns:
            tuple[int, int]: The index of the Sh and Sg combination in the list of combinations.

        """
        return (self.sh_combination.sequence_nr, self.sg_combination.sequence_nr)
