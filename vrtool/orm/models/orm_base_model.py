from peewee import Model

from vrtool.orm.orm_db import vrtool_db

# The following properties and methods are used accross the whole `orm.models`
# subproject
_max_char_length = 128


def _get_table_name(qual_name: str) -> str:
    """
    When invoking the Meta inner class we can access the `__qual__` attribute
    which contains its parent class with the name to be used as a SQLite table

    Args:
        qual_name (str): Value of the `__qual__` attribute.

    Returns:
        str: Name of the table.
    """
    return qual_name.split(".")[0]


class OrmBaseModel(Model):
    """
    Base class meant to gather all common definitions which should be shared
    across all different models of our ORM.
    """

    class Meta:
        """
        Define which database will be coupled to all models inherting from this.
        """

        database = vrtool_db
