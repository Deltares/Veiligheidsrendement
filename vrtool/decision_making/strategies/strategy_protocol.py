from pathlib import Path
from typing import Protocol

from typing_extensions import runtime_checkable

from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.section_as_input import SectionAsInput


@runtime_checkable
class StrategyProtocol(Protocol):
    # In `StrategyBase` this is `type` but we want to avoid using "protected" names
    # for our own properties / attributes.
    design_method: str

    def evaluate(self, traject: DikeTraject, sections: list[SectionAsInput], **kwargs):
        """
        Evaluates the provided measures.
        TODO: For now the arguments are not specific as we do not have a clear view
        on which generic input structures will be used across the different strategies.
        """
        pass

    def make_solution(self, csv_path, step=False, type="Final"):
        """This is a routine to write the results for different types of solutions. It provides a dataframe with for each section the final measure.
        There are 3 types:
        FinalSolution: which is the result in the last step of the optimization
        OptimalSolution: the result with the lowest total cost
        SatisfiedStandardSolution: the result at which the reliability requirement is met.
        Note that if type is not Final the step parameter has to be defined.
        """
        pass

    def determine_risk_cost_curve(self, flood_damage: float, output_path: Path):
        pass

    def get_total_lcc_and_risk(self, step_number: int) -> tuple[float, float]:
        pass
