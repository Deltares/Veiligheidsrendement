import pandas as pd

from vrtool.decision_making.measures.modified_dike_geometry_measure_input import (
    ModifiedDikeGeometryMeasureInput,
)


class TestModifiedDikeGeometryOutput:
    def test_from_dictionary_returns_expected_output(self):
        # Setup
        new_geometry = pd.Series()
        area_extra = 1.1
        area_excavated = 2.2
        d_house = 3.3
        d_crest = 4.4
        d_berm = 5.5

        measure_input = {
            "modified_geometry": new_geometry,
            "area_extra": area_extra,
            "area_excavated": area_excavated,
            "d_house": d_house,
            "d_crest": d_crest,
            "d_berm": d_berm,
            "id": 1,
        }

        # Call
        output = ModifiedDikeGeometryMeasureInput(**measure_input)

        # Assert
        assert output.modified_geometry is new_geometry
        assert output.area_extra == area_extra
        assert output.area_excavated == area_excavated
        assert output.d_house == d_house
        assert output.d_crest == d_crest
        assert output.d_berm == d_berm
