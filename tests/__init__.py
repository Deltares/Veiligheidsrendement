import shutil
from pathlib import Path

from pytest import FixtureRequest

test_data = Path(__file__).parent / "test_data"
test_results = Path(__file__).parent / "test_results"
test_externals = Path(__file__).parent / "test_externals"

if not test_results.is_dir():
    test_results.mkdir(parents=True)


def get_test_results_dir(request: FixtureRequest) -> Path:
    _test_dir: Path = test_results / request.node.originalname
    if _test_dir.is_dir():
        shutil.rmtree(_test_dir)
    _test_dir.mkdir(parents=True)
    return _test_dir
