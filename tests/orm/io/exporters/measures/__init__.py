from typing import Type
import pandas as pd
import pytest

from tests.orm import get_basic_measure_per_section
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.measure_per_section import MeasurePerSection


def create_section_reliability(years: list[int]) -> SectionReliability:
    _section_reliability = SectionReliability()

    _section_reliability.SectionReliability = pd.DataFrame.from_dict(
        {
            "IrrelevantMechanism1": [year / 12.0 for year in years],
            "IrrelevantMechanism2": [year / 13.0 for year in years],
            "Section": [year / 10.0 for year in years],
        },
        orient="index",
        columns=years,
    )
    return _section_reliability


class MeasureWithDictMocked(MeasureProtocol):
    def __init__(self, cost: float, section_reliability: SectionReliability) -> None:
        self.measures = {
            "Cost": cost,
            "Reliability": section_reliability,
        }


class MeasureWithListOfDictMocked(MeasureProtocol):
    def __init__(self, cost: float, section_reliability: SectionReliability) -> None:
        self.measures = [
            {
                "Cost": cost,
                "Reliability": section_reliability,
                "id": "Mocked Dict",
            }
        ]


class MeasureWithMeasureResultCollectionMocked(MeasureProtocol):
    def __init__(self, cost: float, section_reliability: SectionReliability) -> None:
        class MeasureResultCollectionMocked(MeasureResultCollectionProtocol):
            def __init__(self) -> None:
                class MeasureResultMocked(MeasureResultProtocol):
                    def __init__(self) -> None:
                        self.cost = cost
                        self.section_reliability = section_reliability
                        self.measure_id = "Mocked MeasureResult"

                self.result_collection = [MeasureResultMocked()]

        self.measures = MeasureResultCollectionMocked()


class MeasureResultTestInputData:
    t_columns: list[int]
    expected_cost: float
    section_reliability: SectionReliability
    measure_per_section: MeasurePerSection
    measure: MeasureProtocol

    def __init__(self) -> None:
        self.t_columns = [0, 2, 4, 24, 42]
        self.expected_cost = 42.24
        self.section_reliability = create_section_reliability(self.t_columns)
        self.measure_per_section = get_basic_measure_per_section()

    @classmethod
    def with_measures_type(cls, type_measure: Type[MeasureProtocol]):
        _this = cls()
        _this.measure = type_measure(_this.expected_cost, _this.section_reliability)
        _this.measure.parameters = {"ID": _this.measure_per_section.get_id()}
        return _this
