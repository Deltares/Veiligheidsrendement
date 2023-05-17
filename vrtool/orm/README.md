# Object Relational Mapping

This module focuses on mapping the input database tables (in SQLite) to an object representation within the `vrtool`.

## Module structure.

* Importers: The classes responsible to read one or many tables from a database and trigger the corresponding mappings to `vrtool` objects. They are located in the `vrtool.orm.importers` module.
* Exporters: The classes responsible to write in one or many tables of a database with the current status of our `vrtool` objects. They are located in the `vrtool.orm.exporters` module.
* Models: `Python` objects representing the database entities and their relationships. They are located in the `vrtool.orm.models` module.
* Controllers: A series of endpoints to trigger different actions related to read or write from / to the database. For now located in the `vrtool.orm.orm_controllers.py` file.
* `orm_db.py`. File containing the simple definition of the current (`SQLite` database).

## How to use it?

This module is meant to be used locally. However, it is technical possible to generate a database when using the tool as a sandbox. A simple example can be shown below:

```python
from vrtool.orm.orm_controllers import initialize_database
from pathlib import Path

_my_database_location = Path("C:\\my_repo\\my_database.db")
initialize_database(_my_database_location)
```

It is also possible to load an existing database:

```python
from vrtool.orm.orm_controllers import open_database
from pathlib import Path

_my_database_location = Path("C:\\my_repo\\my_database.db")
open_database(_my_database_location)
## Database integration.
```

To achieve a correct integration with / from the database, we will be using the `peewee` library, which is MIT licensed. You may find more about it in the [peewee GitHub repo](https://github.com/coleifer/peewee).

We make the mappings based on the documentation's diagram:

![VrToolDbEntityDiagram](../../docs/db_diagram/vrtool_sql_input.drawio.png)

We know that some of the properties represented as `int` are actually `booleans`, this will be represented in the python classes. For the rest, we will follow a natural translation:

| SQLite Type | ORM Type | Python Type | Remarks |
| --- | --- | --- | --- |
| text | CharField | str | Max 128 characters|
| text | TextField | str | For large pieces of text (>128 characters)|
| int / integer | IntegerField | int | |
| int / integer | BooleanField | bool | Only on given occassions. |
| real | FloatField | float | |