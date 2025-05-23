import pandas as pd

from tests.orm.conftest import (
    _get_basic_measure_per_section,
    _get_domain_basic_dike_section,
)
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.measures.measure_result_type_converter import (
    filter_supported_parameters_dict,
)
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


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
    """
    This mocked class represents a measure whose `measures` property are a
    `MeasureResultCollectionProperty` type, such as `RevetmentMeasure`.
    """

    def __init__(
        self, measure_parameters: dict, measure_result_parameters: dict
    ) -> None:

        # Definition of internal mocked classes.
        class MeasureResultCollectionMocked(MeasureResultCollectionProtocol):
            def __init__(self) -> None:
                class MeasureResultMocked(MeasureResultProtocol):
                    def __init__(self) -> None:
                        self.cost = measure_result_parameters.pop("Cost")
                        self.section_reliability = measure_result_parameters.pop(
                            "Reliability"
                        )
                        self.measure_id = measure_result_parameters.pop("ID")
                        self._result_parameters = filter_supported_parameters_dict(
                            measure_result_parameters
                        )

                    def get_measure_result_parameters(self) -> list[dict]:
                        return self._result_parameters

                self.result_collection = [MeasureResultMocked()]

        # Keep in order because we 'pop out' the parameters during mocking.
        self.parameters = measure_parameters
        self.measures = MeasureResultCollectionMocked()


class MeasureResultTestInputData:
    t_columns: list[int]
    expected_cost: float
    section_reliability: SectionReliability
    measure_per_section: MeasurePerSection
    measure: MeasureProtocol
    available_mechanisms: list[MechanismEnum]
    domain_dike_section: DikeSection
    parameters_to_validate: dict

    @staticmethod
    def create_section_reliability(years: list[int]) -> SectionReliability:
        _section_reliability = SectionReliability()

        _section_reliability.SectionReliability = pd.DataFrame.from_dict(
            {
                MechanismEnum.OVERFLOW.name: [year / 12.0 for year in years],
                MechanismEnum.STABILITY_INNER.name: [year / 13.0 for year in years],
                "Section": [year / 10.0 for year in years],
            },
            orient="index",
            columns=years,
        )
        return _section_reliability

    @staticmethod
    def create_mechanism_per_section(section_data: SectionData) -> list[MechanismEnum]:
        def create_combination(mechanism: MechanismEnum):
            _mech_inst = Mechanism.create(name=mechanism.name)
            MechanismPerSection.create(section=section_data, mechanism=_mech_inst)

        _mechanisms = [MechanismEnum.OVERFLOW, MechanismEnum.STABILITY_INNER]
        list(map(create_combination, _mechanisms))
        return _mechanisms

    def __init__(self) -> None:
        self.t_columns = [0, 2, 4, 24, 42]
        self.expected_cost = 42.24
        self.section_reliability = self.create_section_reliability(self.t_columns)
        self.measure_per_section = _get_basic_measure_per_section()
        self.available_mechanisms = self.create_mechanism_per_section(
            self.measure_per_section.section.get()
        )
        self.domain_dike_section = _get_domain_basic_dike_section()
        self.parameters_to_validate = dict()

    @classmethod
    def with_measures_type(cls, type_measure: type[MeasureProtocol], parameters: dict):
        _this = cls()
        _this.parameters_to_validate = parameters
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


def validate_clean_database():
    assert not any(MeasureResult.select())
    assert not any(MeasureResultSection.select())
    assert not any(MeasureResultMechanism.select())
    assert not any(MeasureResultParameter.select())


def validate_no_parameters(input_data: MeasureResultTestInputData):
    # Verify no parameters (except ID) are present as input data.
    assert not any(filter(lambda x: x != "ID", input_data.measure.parameters.keys()))


def validate_measure_result_export(
    input_data: MeasureResultTestInputData, parameters_to_validate: dict
):
    # Validate number of entries.
    assert len(MeasureResult.select()) == 1
    _measure_result = MeasureResult.get()
    validate_measure_result_parameters(_measure_result, parameters_to_validate)

    assert len(MeasureResultSection.select()) == len(input_data.t_columns)
    assert len(MeasureResultMechanism.select()) == len(input_data.t_columns) * len(
        input_data.available_mechanisms
    )

    # Validate values.
    for year in input_data.t_columns:
        validate_measure_result_section_year(_measure_result, input_data, year)
        validate_measure_result_mechanisms_year(_measure_result, input_data, year)


def validate_measure_result_section_year(
    measure_result: MeasureResult,
    input_data: MeasureResultTestInputData,
    year: int,
):
    _retrieved_result_section = MeasureResultSection.get_or_none(
        (MeasureResultSection.measure_result == measure_result)
        & (MeasureResultSection.time == year)
    )

    assert isinstance(_retrieved_result_section, MeasureResultSection)
    assert (
        _retrieved_result_section.beta
        == input_data.section_reliability.SectionReliability.loc["Section"][year]
    )
    assert _retrieved_result_section.cost == input_data.expected_cost


def validate_measure_result_mechanisms_year(
    measure_result: MeasureResult,
    input_data: MeasureResultTestInputData,
    year: int,
):
    for _mechanism in input_data.available_mechanisms:
        _mech_inst = Mechanism.get_or_none(Mechanism.name == _mechanism.name)
        _retrieved_result_section = (
            MeasureResultMechanism.select()
            .join(MechanismPerSection)
            .where(
                (MeasureResultMechanism.measure_result == measure_result)
                & (MeasureResultMechanism.time == year)
                & (MeasureResultMechanism.mechanism_per_section.mechanism == _mech_inst)
            )
            .get_or_none()
        )

        assert isinstance(_retrieved_result_section, MeasureResultMechanism)
        assert (
            _retrieved_result_section.beta
            == input_data.section_reliability.SectionReliability.loc[_mechanism.name][
                year
            ]
        )


def validate_measure_result_parameters(
    measure_result: MeasureResult, parameters_to_validate: dict
):
    def measure_result_parameter_exists(name: str, value: float) -> bool:
        return (
            MeasureResultParameter.select()
            .where(
                (MeasureResultParameter.name == name.upper())
                & (MeasureResultParameter.value == value)
                & (MeasureResultParameter.measure_result == measure_result)
            )
            .exists()
        )

    assert len(MeasureResultParameter.select()) == len(parameters_to_validate)
    assert all(
        measure_result_parameter_exists(name, value)
        for name, value in parameters_to_validate.items()
    )
