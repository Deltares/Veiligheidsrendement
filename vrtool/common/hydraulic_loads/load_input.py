import openturns as ot


class LoadInput:
    # class to store load data
    load_type: str
    distribution: dict[int, ot.Distribution]

    def __init__(self, section_fields: list[str]):
        self.load_type = ""
        self.input = {}
        if "Load_2025" in section_fields:
            self.load_type = "HRING"
        elif "YearlyWLRise" in section_fields:
            self.load_type = "SAFE"
