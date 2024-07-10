import shutil
from pathlib import Path
from typing import Iterator

import pytest

from tests import test_data, test_results
from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database
from vrtool.orm.version.migration.database_version import DatabaseVersion
from vrtool.orm.version.orm_version import OrmVersion


@pytest.fixture(name="empty_db_path")
def get_empty_db_path_fixture(request: pytest.FixtureRequest) -> Iterator[Path]:
    """
    Gets an empty database path with a valid scheme.
    """
    # Create a results directory where to persist the database.
    _output_dir = test_results.joinpath(request.node.name)
    if _output_dir.exists():
        shutil.rmtree(_output_dir)
    _output_dir.mkdir(parents=True)
    _test_db_file = _output_dir.joinpath("test_db.db")

    # Copy the original `empty_db.db` into the output directory.
    _db_file = test_data.joinpath("test_db", "empty_db.db")
    shutil.copyfile(_db_file, _test_db_file)

    yield _test_db_file

    _test_db_file.unlink(missing_ok=True)


@pytest.fixture(name="valid_conversion_input")
def get_valid_conversion_input_fixture(
    empty_db_path: Path,
    request: pytest.FixtureRequest,
) -> Iterator[Path]:
    def generate_script_name(version: DatabaseVersion) -> str:
        return f"v{version.major}_{version.minor}_{version.patch}.sql"

    def drop_table_script(version: DatabaseVersion):
        _drop_script = _input_dir.joinpath(generate_script_name(version))
        _drop_script.write_text(
            """
            -- This script will not be executed as it has the same version as the ORM
            DROP TABLE IF EXISTS Version;
            """
        )

    def insert_script(version: DatabaseVersion):
        _insert_script = _input_dir.joinpath(generate_script_name(version))
        _insert_script.write_text(
            f"""
            INSERT INTO Version (orm_version) VALUES ('{version}');
            """
        )

    def valid_script():
        _valid_script = _input_dir.joinpath("valid_script.txt")
        _valid_script.write_text(
            """
            CREATE TABLE IF NOT EXISTS ValidTable (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            );
            """
        )

    def wrong_script():
        _wrong_script = _input_dir.joinpath("wrong_script.txt")
        _wrong_script.write_text("THIS IS NO SQL!")

    def prepare_database(database_version: DatabaseVersion):
        with open_database(database_version.database_path).connection_context():
            _version, _ = DbVersion.get_or_create()
            _version.orm_version = str(database_version)
            _version.save()

    # Remove existing scripts
    _input_dir = test_results.joinpath(request.node.name)
    for _script in _input_dir.rglob("*.sql"):
        _script.unlink()

    # Set the database version to the current ORM version
    _orm_version = OrmVersion.from_orm()
    _db_version = DatabaseVersion(**_orm_version.__dict__, database_path=empty_db_path)
    prepare_database(_db_version)

    # Create migration scripts
    drop_table_script(_db_version)
    _db_version.patch += 1
    insert_script(_db_version)
    _db_version.minor += 1
    insert_script(_db_version)
    _db_version.major += 1
    insert_script(_db_version)
    _db_version.patch += 1
    insert_script(_db_version)

    valid_script()
    wrong_script()

    yield _input_dir

    shutil.rmtree(_input_dir)
