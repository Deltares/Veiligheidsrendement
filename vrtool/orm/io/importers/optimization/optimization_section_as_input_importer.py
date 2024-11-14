import math
from collections import defaultdict

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.orm.io.importers.optimization.optimization_measure_result_importer import (
    OptimizationMeasureResultImporter,
)
from vrtool.orm.models.measure_result import MeasureResult as OrmMeasureResult
from vrtool.orm.models.section_data import SectionData as OrmSectionData
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class OptimizationSectionAsInputImporter:
    config: VrtoolConfig

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        self.config = vrtool_config

    def _import_initial_assessment(
        self, section_data: OrmSectionData
    ) -> MechanismPerYearProbabilityCollection:
        _mech_collection = MechanismPerYearProbabilityCollection([])
        for _ms in section_data.mechanisms_per_section:
            _mech_enum = MechanismEnum.get_enum(_ms.mechanism.name)
            for _amr in _ms.assessment_mechanism_results:
                _mech_collection.probabilities.append(
                    MechanismPerYear(
                        mechanism=_mech_enum,
                        year=_amr.time,
                        probability=beta_to_pf(_amr.beta),
                    )
                )
        return _mech_collection

    def import_from_section_data_results(
        self,
        section_data_results: tuple[OrmSectionData, dict[OrmMeasureResult, list[int]]],
    ) -> SectionAsInput:
        """
        Imports the filtered collection of `SectionAsInput` containing all the information related
        to the requested measure results (`MeasureResult`).

        Args:
            section_data_results (tuple[OrmSectionData, dict[OrmMeasureResult, list[int]]]): Measure results and requested years to import for a given `SectionData`.

        Returns:
            SectionAsInput: Mapped resulting object.
        """
        _section_imported_measures: list[MeasureAsInputProtocol] = []
        _section_data, _measure_results_dict = section_data_results

        for _measure_result, _investment_years in _measure_results_dict.items():
            _imported_measures = OptimizationMeasureResultImporter(
                self.config, _investment_years
            ).import_orm(_measure_result)
            _section_imported_measures.extend(_imported_measures)

        def filter_by_type(
            measure_type: type[MeasureAsInputProtocol],
        ) -> list[MeasureAsInputProtocol]:
            return list(
                filter(
                    lambda x: isinstance(x, measure_type), _section_imported_measures
                )
            )

        def set_base_cost(measure_type: type[MeasureAsInputProtocol]):
            """
            Sets the base costs by clustering the measures by both type and
            `l_stab_screen` value.
            Args:
                measure_type (type[MeasureAsInputProtocol]): Measure type to use
                for measure filtering.
            """
            _base_costs = defaultdict(lambda: defaultdict(lambda: 0.0))
            _measure_collection = filter_by_type(measure_type)

            # Base costs are the `cost` of an "initial measure".
            def valid_l_stab_screen(l_stab_screen: float) -> float:
                return -1 if math.isnan(l_stab_screen) else l_stab_screen

            for _initial_measure in filter(
                lambda x: x.is_base_measure(),
                _measure_collection,
            ):
                # We pivot by both `type[MeasureAsInputProtocol]` and
                # `l_stab_screen`, as for different values of the latter
                # you might find different "initial" measures.
                _base_costs[_initial_measure.measure_type][
                    valid_l_stab_screen(_initial_measure.l_stab_screen)
                ] = _initial_measure.cost

            for _measure in _measure_collection:
                _measure.base_cost = _base_costs[_measure.measure_type][
                    valid_l_stab_screen(_measure.l_stab_screen)
                ]

        # TODO: Do we need to set base costs for `SgMeasure` type?
        set_base_cost(SgMeasure)
        set_base_cost(ShMeasure)

        return SectionAsInput(
            section_name=_section_data.section_name,
            traject_name=_section_data.dike_traject.traject_name,
            section_length=_section_data.section_length,
            measures=_section_imported_measures,
            flood_damage=_section_data.dike_traject.flood_damage,
            initial_assessment=self._import_initial_assessment(_section_data),
        )
