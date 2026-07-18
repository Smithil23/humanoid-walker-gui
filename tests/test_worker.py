"""Tests that exercise the pure-Python parts (no MATLAB engine required)."""

import numpy as np

from app.matlab_worker import EngineController, _to_np


def test_to_np_handles_none_and_lists():
    assert _to_np(None).size == 0
    out = _to_np([[1.0], [2.0], [3.0]])
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_read_progress_parses_csv(tmp_path):
    ctrl = EngineController(str(tmp_path))
    ctrl.progress_file = str(tmp_path / "progress.csv")
    with open(ctrl.progress_file, "w") as fh:
        fh.write("0,-120.5\n1,-340.0\n2,-512.25\n")

    gens, best = ctrl.read_progress()
    assert gens == [0, 1, 2]
    assert np.allclose(best, [-120.5, -340.0, -512.25])


def test_read_progress_missing_file_is_empty(tmp_path):
    ctrl = EngineController(str(tmp_path))
    ctrl.progress_file = str(tmp_path / "does_not_exist.csv")
    gens, best = ctrl.read_progress()
    assert gens == []
    assert best == []


def test_stop_flag_roundtrip(tmp_path):
    ctrl = EngineController(str(tmp_path))
    ctrl.stop_file = str(tmp_path / "stop.flag")
    ctrl.request_stop()
    assert __import__("os").path.exists(ctrl.stop_file)
    ctrl._clear_stop()
    assert not __import__("os").path.exists(ctrl.stop_file)
