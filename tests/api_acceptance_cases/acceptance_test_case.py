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

    case_name: str
    model_directory: str
    traject_name: str
    excluded_mechanisms: list[MechanismEnum] = field(
        default_factory=lambda: [
            MechanismEnum.HYDRAULIC_STRUCTURES,
        ]
    )

    @staticmethod
    def get_cases() -> list[AcceptanceTestCase]:
        # Defining acceptance test cases so they are accessible from the other test classes.
        return [
            AcceptanceTestCase(
                model_directory="31-1 two coastal sections",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, two coastal sections",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two river sections",
            ),
            AcceptanceTestCase(
                model_directory="31-1 base coastal case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, base coastal case",
            ),
            AcceptanceTestCase(
                model_directory="38-1 base river case",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, base river case",
            ),
            AcceptanceTestCase(
                model_directory="31-1 mixed coastal case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, mixed coastal case",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections D-Stability",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two sections with D-Stability",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections anchored sheetpile",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two river sections with anchored sheetpile [VRTOOL-344]",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections beta piping input",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two river sections with beta piping input",
            ),
            AcceptanceTestCase(
                model_directory="38-1 custom measures",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, custom Measures, MVP",
            ),
            AcceptanceTestCase(
                model_directory="38-1 custom measures high betas low costs",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, custom measures with high betas and low costs",
            ),
            AcceptanceTestCase(
                model_directory="38-1 custom measures low betas high costs",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, custom measures with low betas and high costs",
            ),
            AcceptanceTestCase(
                model_directory="38-1 custom measures real cases",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, custom measures with real cases",
            ),
        ]
