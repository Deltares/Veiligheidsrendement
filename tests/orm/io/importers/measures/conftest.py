from vrtool.common.enums import MechanismEnum
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.custom_measure_detail import CustomMeasureDetail
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.standard_measure import StandardMeasure


def set_standard_measure(measure: Measure) -> None:
    StandardMeasure.create(
        measure=measure,
        crest_step=4.2,
        direction="onwards",
        stability_screen=False,
        max_crest_increase=0.1,
        max_outward_reinforcement=2,
        max_inward_reinforcement=3,
        prob_of_solution_failure=0.4,
        failure_probability_with_solution=0.5,
    )


def set_custom_measure(measure: Measure) -> None:
    _mech_inst = Mechanism.create(name=MechanismEnum.INVALID.name)
    _measure_per_section, _ = MeasurePerSection.get_or_create(
        measure=measure, section=SectionData.get()
    )
    _mechanism_per_section, _ = MechanismPerSection.get_or_create(
        mechanism=_mech_inst, section=_measure_per_section.section
    )
    CustomMeasureDetail.create(
        measure=measure,
        mechanism_per_section=_mechanism_per_section,
        cost=1234.56,
        beta=42.24,
        year=2023,
    )


def get_valid_measure(
    measure_type: MeasureTypeEnum,
    combinable_type: CombinableTypeEnum,
) -> Measure:
    _measure_type = MeasureType.create(name=measure_type.legacy_name)
    _combinable_type = CombinableType.create(name=combinable_type.name)
    _measure = Measure.create(
        measure_type=_measure_type,
        combinable_type=_combinable_type,
        name="Test Measure",
        year=2023,
    )
    if measure_type == MeasureTypeEnum.CUSTOM:
        set_custom_measure(_measure)
    else:
        set_standard_measure(_measure)
    return _measure
