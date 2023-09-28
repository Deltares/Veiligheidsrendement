from tests.orm.io.exporters.measures import MeasureResultTestInputData
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

def validate_clean_database():
    assert not any(MeasureResult.select())
    assert not any(MeasureResultSection.select())
    assert not any(MeasureResultMechanism.select())
    assert not any(MeasureResultParameter.select())

def validate_no_parameters(input_data: MeasureResultTestInputData):
    # Verify no parameters (except ID) are present as input data.
        assert not any(
            filter(lambda x: x != "ID", input_data.measure.parameters.keys())
        )

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
    for _mechanism_name in input_data.available_mechanisms:
        _mechanism = Mechanism.get_or_none(Mechanism.name == _mechanism_name)
        _retrieved_result_section = (
            MeasureResultMechanism.select()
            .join(MechanismPerSection)
            .where(
                (MeasureResultMechanism.measure_result == measure_result)
                & (MeasureResultMechanism.time == year)
                & (MeasureResultMechanism.mechanism_per_section.mechanism == _mechanism)
            )
            .get_or_none()
        )

        assert isinstance(_retrieved_result_section, MeasureResultMechanism)
        assert (
            _retrieved_result_section.beta
            == input_data.section_reliability.SectionReliability.loc[_mechanism_name][
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
