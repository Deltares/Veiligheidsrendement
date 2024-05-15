import openturns as ot


class LoadInput:
    # class to store load data
    load_type: str

    def __init__(self, section_fields: list[str]):
        self.load_type = ""
        self.input = {}
        if "Load_2025" in section_fields:
            self.load_type = "HRING"
        elif "YearlyWLRise" in section_fields:
            self.load_type = "SAFE"

    def set_annual_change(
        self, change_type: str = "determinist", parameters: list[float] = [0]
    ):
        # set an annual change of the water level
        if change_type == "determinist":
            self.dist_change = ot.Dirac(parameters)
        elif change_type == "SAFE":  # specific formulation for SAFE
            self.dist_change = parameters[0]
            self.HBN_factor = parameters[1]
        elif change_type == "gamma":
            self.dist_change = ot.Gamma()
            self.dist_change.setParameter(ot.GammaMuSigma()(parameters))
