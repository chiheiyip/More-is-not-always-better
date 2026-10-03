import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("teacher_publisher", Path(__file__).resolve().parents[1] / "scripts/publish_latest_teacher_results.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.mark.parametrize("flip,direction,status", [
    (False, float("nan"), "未改变"), (True, float("nan"), "改变"),
    (False, True, "改变"), (False, False, "未改变"),
    ("False", None, "未改变"), ("True", None, "改变"),
])
def test_omnibus_direction_is_not_a_change(flip, direction, status):
    assert publisher.change_status(flip, direction) == status
