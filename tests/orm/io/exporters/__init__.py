from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


def create_required_mechanism_per_section(
    section_data: SectionData, mechanism_available_list: list[MechanismEnum]
) -> None:
    _added_mechanisms = []
    for mechanism in mechanism_available_list:
        _mech_inst, _ = Mechanism.get_or_create(name=mechanism.name)
        _added_mechanisms.append(
            MechanismPerSection.create(section=section_data, mechanism=_mech_inst)
        )
