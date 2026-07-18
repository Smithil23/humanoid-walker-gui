"""
MATLAB engine management and background workers for the Humanoid Walker GUI.

Design notes
------------
* The MATLAB engine is started ONCE (it is slow, ~10 s) on a background
  thread, then reused for every run.
* All engine calls happen on worker threads so the Qt event loop never blocks.
  The MATLAB engine is not safe to call from multiple threads at once, so we
  keep to a single worker at a time (the GUI disables Run while one is active).
* GA progress is streamed: the MATLAB OutputFcn appends "gen,bestFitness" to a
  CSV, and the GUI polls that file with a QTimer. The worker itself just blocks
  on the (long) engine call until it returns.
* RL has no per-episode Python hook, so its live view is MATLAB's own Episode
  Manager; the worker returns the full episode-reward vector at the end.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal


def _to_np(x):
    """Convert a matlab.double (or None/empty) to a flat numpy array."""
    if x is None:
        return np.array([])
    arr = np.array(x, dtype=float)
    return arr.flatten()


class EngineStarter(QObject):
    """Starts the MATLAB engine on a worker thread."""

    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, project_dir: str):
        super().__init__()
        self.project_dir = project_dir
        self.eng = None

    def run(self):
        try:
            import matlab.engine  # imported here so the GUI can load without it

            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.project_dir, nargout=0)
            # The bridge .m files live in matlab_bridge/ inside THIS repo, i.e.
            # one level up from this file (app/) then into matlab_bridge/.
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bridge = os.path.join(repo_root, "matlab_bridge")
            if os.path.isdir(bridge):
                self.eng.addpath(bridge, nargout=0)
            else:
                self.failed.emit(
                    f"matlab_bridge folder not found at {bridge}. "
                    "Make sure the repo layout is intact."
                )
                return
            self.eng.cd(self.project_dir, nargout=0)
            self.ready.emit(self.eng)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TrainWorker(QObject):
    """Runs one GA or RL job and emits the result."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        eng,
        method: str,
        overrides: dict,
        do_train: bool,
        use_parallel: bool,
        progress_file: str,
        stop_file: str,
        agent_file: str = "",
        rl_model: str = "ddpg",
    ):
        super().__init__()
        self.eng = eng
        self.method = method
        self.overrides = overrides
        self.do_train = do_train
        self.use_parallel = use_parallel
        self.progress_file = progress_file
        self.stop_file = stop_file
        self.agent_file = agent_file
        self.rl_model = rl_model

    def run(self):
        try:
            payload = json.dumps(self.overrides)
            if self.method == "ga":
                best_x, fval, r_time, r_data = self.eng.gui_run_ga(
                    payload,
                    self.progress_file,
                    self.stop_file,
                    self.do_train,
                    nargout=4,
                )
                result = {
                    "method": "ga",
                    "best_x": _to_np(best_x),
                    "fval": float(fval),
                    "reward_time": _to_np(r_time),
                    "reward_data": _to_np(r_data),
                }
            else:  # rl
                ep_reward, r_time, r_data = self.eng.gui_run_rl(
                    payload,
                    self.do_train,
                    self.use_parallel,
                    self.agent_file,
                    self.rl_model,
                    nargout=3,
                )
                result = {
                    "method": "rl",
                    "episode_reward": _to_np(ep_reward),
                    "reward_time": _to_np(r_time),
                    "reward_data": _to_np(r_data),
                }
            # Derived headline metrics both paths share.
            rd = result["reward_data"]
            result["total_reward"] = float(rd.sum()) if rd.size else float("nan")
            result["sim_steps"] = int(rd.size)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class EngineController:
    """Owns the engine + threads and exposes a small API to the GUI."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.eng = None
        self._thread = None
        self._worker = None
        # Temp files for GA streaming + cooperative stop.
        tmp = tempfile.gettempdir()
        self.progress_file = os.path.join(tmp, "hw_ga_progress.csv")
        self.stop_file = os.path.join(tmp, "hw_stop.flag")

    # -- engine startup ---------------------------------------------------
    def start_engine(self, on_ready, on_failed):
        # on_ready(eng) and on_failed(msg) must be bound methods of a QObject
        # living in the GUI thread, so Qt delivers them there (not the worker).
        self._thread = QThread()
        self._starter = EngineStarter(self.project_dir)
        self._starter.moveToThread(self._thread)
        self._thread.started.connect(self._starter.run)
        self._starter.ready.connect(self._thread.quit)
        self._starter.ready.connect(on_ready)
        self._starter.failed.connect(self._thread.quit)
        self._starter.failed.connect(on_failed)
        self._thread.start()

    # -- training ---------------------------------------------------------
    def run_training(
        self,
        method,
        overrides,
        do_train,
        use_parallel,
        on_finished,
        on_failed,
        agent_file="",
        rl_model="ddpg",
    ):
        self._clear_stop()
        # Fresh progress file so the poller starts clean.
        try:
            open(self.progress_file, "w").close()
        except OSError:
            pass

        self._thread = QThread()
        self._worker = TrainWorker(
            self.eng,
            method,
            overrides,
            do_train,
            use_parallel,
            self.progress_file,
            self.stop_file,
            agent_file,
            rl_model,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(on_failed)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def request_stop(self):
        """Signal a cooperative stop (GA checks this in its OutputFcn)."""
        try:
            open(self.stop_file, "w").close()
        except OSError:
            pass

    def _clear_stop(self):
        try:
            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
        except OSError:
            pass

    # -- GA progress polling ---------------------------------------------
    def read_progress(self):
        """Return (generations, best_fitness) lists from the GA CSV so far."""
        gens, best = [], []
        try:
            with open(self.progress_file, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    g, b = line.split(",")
                    gens.append(int(float(g)))
                    best.append(float(b))
        except (OSError, ValueError):
            pass
        return gens, best
