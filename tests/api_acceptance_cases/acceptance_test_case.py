from __future__ import annotations

from dataclasses import dataclass, field

import vrtool.orm.models as orm
from vrtool.common.enums.mechanism_enum import MechanismEnum

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)

vrtool_db_default_name = "vrtool_input.db"


@dataclass
class AcceptanceTestCase:
    """
    Dataclass containing the required information to run one of our acceptance
    test cases, often related to the usage and comparison of databases.
    """

    model_directory: str
    traject_name: str
    excluded_mechanisms: list[MechanismEnum] = field(
        default_factory=lambda: [
            MechanismEnum.HYDRAULIC_STRUCTURES,
        ]
    )
    run_adjusted_timing: bool = False
    run_filtered: bool = False

    @staticmethod
    def get_cases() -> list[AcceptanceTestCase]:
        # Defining acceptance test cases so they are accessible from the other test classes.
        return [
            AcceptanceTestCase(
                model_directory="31-1_two_coastal_sections",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                run_adjusted_timing=True,
                run_filtered=True,
            ),
            AcceptanceTestCase(
                model_directory="38-1_two_river_sections",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                run_adjusted_timing=True,
                run_filtered=True,
            ),
            AcceptanceTestCase(
                model_directory="31-1_base_coastal_case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_base_river_case",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="31-1_mixed_coastal_case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_two_river_sections_D-Stability",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_two_river_sections_anchored_sheetpile",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_custom_measures",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_custom_measures_high_betas_low_costs",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_custom_measures_low_betas_high_costs",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
            AcceptanceTestCase(
                model_directory="38-1_custom_measures_real_cases",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
            ),
        ]
