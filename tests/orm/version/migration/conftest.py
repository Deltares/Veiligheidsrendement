import shutil
from pathlib import Path
from typing import Iterator

import pytest

from tests import test_data, test_results
from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database_without_compatibility_check
from vrtool.orm.version.orm_version import OrmVersion


@pytest.fixture(name="output_dir", scope="module")
def _get_output_dir_fixture(request: pytest.FixtureRequest) -> Iterator[Path]:
    _output_dir = test_results.joinpath(request.node.name)
    if _output_dir.exists():
        shutil.rmtree(_output_dir)
    _output_dir.mkdir(parents=True)

    yield _output_dir

    shutil.rmtree(_output_dir)


@pytest.fixture(name="valid_conversion_scripts", scope="module")
def get_valid_conversion_scripts_fixture(
    output_dir: Path,
) -> Iterator[Path]:
    """
    Get valid conversion scripts for the migration tests.

    Args:
        output_dir (Path): Output directory for the scripts.

    Yields:
        Iterator[Path]: Path to the valid conversion scripts.
    """

    def generate_script_name(version: OrmVersion) -> str:
        return f"v{version.major}_{version.minor}_{version.patch}.sql"

    def drop_table_script(version: OrmVersion):
        _drop_script = output_dir.joinpath(generate_script_name(version))
        _drop_script.write_text(
            """
            -- This script will not be executed as it has the same version as the ORM
            DROP TABLE IF EXISTS Version;
            """
        )

    def insert_script(version: OrmVersion):
        _insert_script = output_dir.joinpath(generate_script_name(version))
        _insert_script.write_text(
            f"""
            INSERT INTO Version (orm_version) VALUES ('{version}');
            """
        )

    def valid_script():
        _valid_script = output_dir.joinpath("valid_script.txt")
        _valid_script.write_text(
            """
            CREATE TABLE IF NOT EXISTS ValidTable (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            );
            """
        )

    def wrong_script():
        _wrong_script = output_dir.joinpath("wrong_script.txt")
        _wrong_script.write_text("THIS IS NO SQL!")

    # Set the database version to the current ORM version
    _orm_version = OrmVersion.from_orm()

    # Create migration scripts
    drop_table_script(_orm_version)
    _orm_version.patch += 1
    insert_script(_orm_version)
    _orm_version.minor += 1
    insert_script(_orm_version)
    _orm_version.major += 1
    insert_script(_orm_version)
    _orm_version.patch += 1
    insert_script(_orm_version)

    valid_script()
    wrong_script()

    yield output_dir


@pytest.fixture(name="empty_db_path", scope="module")
def _get_empty_database_fixture(output_dir: Path) -> Iterator[Path]:
    # Copy the original `empty_db.db` into the output directory.
    _db_file = test_data.joinpath("test_db", "empty_db.db")
    _test_db_file = output_dir.joinpath("test_db.db")
    shutil.copyfile(_db_file, _test_db_file)

    yield _test_db_file


@pytest.fixture(name="valid_conversion_db")
def get_valid_conversion_db_fixture(
    empty_db_path: Path,
) -> Iterator[Path]:
    """
    Get a valid database for conversion based on the empty database
    with the current ORM version.

    Args:
        output_dir (Path): Output directory for the database.

    Yields:
        Iterator[Path]: Path to the valid database.
    """

    # Set the database version to the current ORM version
    _orm_version = OrmVersion.from_orm()

    with open_database_without_compatibility_check(empty_db_path).connection_context():
        _version, _ = DbVersion.get_or_create()
        _version.orm_version = str(_orm_version)
        _version.save()

    yield empty_db_path
