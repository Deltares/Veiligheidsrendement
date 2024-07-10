from pathlib import Path

from vrtool.orm.version.migration.script_version import ScriptVersion


class TestScriptVersion:
    def test_from_script(self):
        # 1. Define test data
        _script_path = Path("v1_2_3.sql")

        # 2. Execute test
        _script_version = ScriptVersion.from_script(_script_path)

        # 3. Verify expectations
        assert isinstance(_script_version, ScriptVersion)
        assert str(_script_version) == "1.2.3"
        assert _script_version.script_path == _script_path
