from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm import orm_models
from vrtool.flood_defence_system.dike_section import DikeSection

def import_dike_traject_info(orm_dike_traject: orm_models.DikeTrajectInfo) -> DikeTrajectInfo:
    if not orm_dike_traject:
        raise ValueError(f"No valid value given for {orm_models.DikeTrajectInfo.__name__}.")

    return DikeTrajectInfo(
        traject_name=orm_dike_traject.traject_name,
        omegaPiping=orm_dike_traject.omega_piping,
        omegaStabilityInner=orm_dike_traject.omega_stability_inner,
        omegaOverflow=orm_dike_traject.omega_overflow,
        aPiping=orm_dike_traject.a_piping,
        bPiping=orm_dike_traject.b_piping,
        aStabilityInner=orm_dike_traject.a_stability_inner,
        bStabilityInner=orm_dike_traject.b_stability_inner,
        beta_max=orm_dike_traject.beta_max,
        Pmax=orm_dike_traject.p_max,
        FloodDamage=orm_dike_traject.flood_damage,
        TrajectLength=orm_dike_traject.traject_length
    )

def import_dike_traject(orm_dike_traject_info: orm_models.DikeTrajectInfo) -> DikeTraject:
    _dike_traject = DikeTraject()
    _dike_traject.general_info = import_dike_traject_info(orm_dike_traject_info)

    # Currently it is assumed that all SectionData present in a db belongs to whatever traject name has been provided.
    _dike_traject.sections = import_dike_section_list(orm_dike_traject_info.dike_sections.select().where(orm_models.SectionData.in_analysis == True))

    # _dike_traject.mechanism_names = config.mechanisms
    # _dike_traject.assessment_plot_years = config.assessment_plot_years
    # _dike_traject.flip_traject = config.flip_traject
    # _dike_traject.t_0 = config.t_0
    # _dike_traject.T = config.T

    return _dike_traject

def import_dike_section(orm_dike_section: orm_models.SectionData) -> DikeSection:
    _dike_section = DikeSection()
    _dike_section.name = orm_dike_section.section_name
    _mechanisms = orm_dike_section.mechanisms_per_section.select(orm_models.MechanismPerSection.mechanism)

    return _dike_section


def import_dike_section_list(orm_dike_section_list: list[orm_models.SectionData]) -> list[DikeSection]:
    return list(map(import_dike_section, orm_dike_section_list))