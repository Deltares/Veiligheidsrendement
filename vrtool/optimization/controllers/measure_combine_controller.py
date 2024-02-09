from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sh_measure import ShMeasure


class MeasureCombineController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def combine(self) -> list[CombinedMeasure]:
        _combined_measures = []

        for _primary in self._section.sh_measures:
            if _primary.combine_type != CombinableTypeEnum.PARTIAL:
                continue
            _combined_measures.append(CombinedMeasure(_primary, None))
            for _secondary in self._section.sh_measures:
                if (
                    _secondary.combine_type
                    not in ShMeasure.get_allowed_combinable_types()
                ):
                    continue
                _combined_measures.append(CombinedMeasure(_primary, _secondary))
        for _sg in self._section.sg_measures:
            pass

        return _combined_measures
