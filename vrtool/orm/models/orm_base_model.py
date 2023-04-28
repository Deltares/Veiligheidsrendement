from peewee import Model

from vrtool.orm.orm_db import vrtool_db


def _get_table_name(qual_name: str) -> str:
    """
    When invoking the Meta inner class we can access the `__qual__` attribute which contains its parent class with the name to be used as a SQLite table

    Args:
        qual_name (str): Value of the `__qual__` attribute.

    Returns:
        str: Name of the table.
    """
    return qual_name.split(".")[0]


_max_char_length = 128


class OrmBaseModel(Model):
    class Meta:
        database = vrtool_db
