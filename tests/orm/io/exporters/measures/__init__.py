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
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


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


def create_mechanism_per_section(section_data: SectionData) -> list[str]:
    def create_combination(mechanism_name: str):
        _mechanism = Mechanism.create(name=mechanism_name)
        MechanismPerSection.create(section=section_data, mechanism=_mechanism)

    _mechanism_names = ["IrrelevantMechanism1", "IrrelevantMechanism2"]
    list(map(create_combination, _mechanism_names))
    return _mechanism_names


class MeasureWithDictMocked(MeasureProtocol):
    """
    This mocked class represents a measure whose `measures` property is just a `dict`.
    """

    def __init__(
        self, measure_parameters: dict, measure_result_parameters: dict
    ) -> None:
        self.measures = measure_result_parameters
        self.parameters = measure_parameters


class MeasureWithListOfDictMocked(MeasureProtocol):
    """
    This mocked class represents a measure whose `measures` property are a `list[dict]` type,
    at the moment only present for `SoilReinforcementMeasure`.
    """

    def __init__(
        self, measure_parameters: dict, measure_result_parameters: dict
    ) -> None:
        self.measures = [measure_result_parameters]
        self.parameters = measure_parameters


class MeasureWithMeasureResultCollectionMocked(MeasureProtocol):
    def __init__(
        self, measure_parameters: dict, measure_result_parameters: dict
    ) -> None:
        class MeasureResultCollectionMocked(MeasureResultCollectionProtocol):
            def __init__(self) -> None:
                class MeasureResultMocked(MeasureResultProtocol):
                    def __init__(self) -> None:
                        self.cost = measure_result_parameters.pop("Cost")
                        self.section_reliability = measure_result_parameters.pop(
                            "Reliability"
                        )
                        self.measure_id = measure_result_parameters.pop("ID")
                        self._result_parameters = measure_result_parameters

                    def get_measure_result_parameters(self) -> list[dict]:
                        return self._result_parameters

                self.result_collection = [MeasureResultMocked()]

        self.parameters = measure_parameters
        self.measures = MeasureResultCollectionMocked()


class MeasureResultTestInputData:
    t_columns: list[int]
    expected_cost: float
    section_reliability: SectionReliability
    measure_per_section: MeasurePerSection
    measure: MeasureProtocol
    available_mechanisms: list[str]

    def __init__(self) -> None:
        self.t_columns = [0, 2, 4, 24, 42]
        self.expected_cost = 42.24
        self.section_reliability = create_section_reliability(self.t_columns)
        self.measure_per_section = get_basic_measure_per_section()
        self.available_mechanisms = create_mechanism_per_section(
            self.measure_per_section.section.get()
        )

    @classmethod
    def with_measures_type(cls, type_measure: Type[MeasureProtocol], parameters: dict):
        _this = cls()

        _this.measure = type_measure(
            measure_parameters={"ID": _this.measure_per_section.get_id()},
            measure_result_parameters={
                "ID": _this.measure_per_section.get_id(),
                "Cost": _this.expected_cost,
                "Reliability": _this.section_reliability,
            }
            | parameters,
        )
        return _this
