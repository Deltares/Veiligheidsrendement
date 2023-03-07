import pandas as pd


class DikeProfile:
    """
    Class that contains a DikeProfile consisting of 4 or 6 characteristic points:
    """

    def __init__(self, name: str = None):
        self.characteristic_points = {}
        self.name = name

    def add_point(self, key, xz):
        self.characteristic_points[key] = xz

    def generate_shapely_polygon(self):
        pass

    def read_points(self):
        pass

    def to_csv(self, path):
        # add mkdir?
        pd.DataFrame.from_dict(
            self.characteristic_points, orient="index", columns=["x", "z"]
        ).to_csv(path.joinpath("{}.csv".format(self.name)))
