from pathlib import Path

import click
import pandas

from vrtool.orm.io.exporters.measures.list_of_dict_to_custom_measure_exporter import (
    ListOfDictToCustomMeasureExporter,
)
from vrtool.orm.orm_controllers import open_database


@click.group()
def cli():
    pass


@cli.command(
    name="export_csv_to_database",
    help="Exports a csv with custom measures into a database.",
)
@click.argument("database_file", type=click.Path(exists=True), nargs=1)
@click.argument("csv_file", type=click.Path(exists=True), nargs=1)
def export_csv_to_database(database_file: str, csv_file: str):
    """
    Can be run with `python export_csv_to_database database_file csv_file`
    """
    _csv_custom_measures = pandas.read_csv(csv_file, delimiter=";").to_dict("records")
    _db = open_database(Path(database_file))

    ListOfDictToCustomMeasureExporter(_db).export_dom(_csv_custom_measures)


if __name__ == "__main__":
    export_csv_to_database()
