from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestSectionReliability:
    def test_init_sets_properties(self):
        # Call
        reliability = SectionReliability()

        # Assert
        assert len(reliability.Mechanisms) == 0
