"""
Humanoid Walker control GUI (PySide6).

Left rail  : pick GA or RL, tune the parameters that map to the actual
             MATLAB variables, and Run / Stop.
Right side : headline metrics, a live training-progress plot, and the
             reward-over-time plot for the simulated individual.

Run with:  python -m app.main   (from the humanoid_walker_gui/ folder)

The MATLAB engine and toolboxes are only needed to actually run training;
the window itself opens without them so you can develop the UI offline.
"""

from __future__ import annotations

import os
import sys

# Force matplotlib to use the SAME Qt binding as the app (PySide6). Without
# this, matplotlib may pick another installed binding (e.g. PyQt5) and its
# canvas widget won't be accepted by PySide6 layouts.
os.environ.setdefault("QT_API", "PySide6")

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.matlab_worker import EngineController

# --------------------------------------------------------------------------
# Point this at your example folder (the one containing sm_humanoid_walker_*).
# --------------------------------------------------------------------------
PROJECT_DIR = os.environ.get(
    "HW_PROJECT_DIR",
    os.path.expanduser(r"~/Desktop/Matlab_Humanoid/TrainHumanoidWalkerExample"),
)


def hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def dspin(lo, hi, val, step, decimals=3):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    return w


def ispin(lo, hi, val, step=1):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    return w


class RewardWeights(QGroupBox):
    """The five reward weights, each with its own realistic range/step.

    Defaults match sm_humanoid_walker_*_parameters.m. Note the very different
    magnitudes: power penalty is ~5e-4 while vertical penalty is 25.
    """

    def __init__(self, lateral_default=2.5):
        super().__init__("Reward weights")
        form = QFormLayout(self)
        self.forward = dspin(0.0, 5.0, 1.0, 0.1, 2)  # w1
        self.timestep = dspin(0.0, 5.0, 1.0, 0.1, 2)  # w2
        self.power = dspin(0.0, 2e-3, 5e-4, 1e-4, 5)  # w3
        self.vertical = dspin(0.0, 50.0, 25.0, 1.0, 1)  # w4
        self.lateral = dspin(0.0, 10.0, lateral_default, 0.1, 2)  # w5
        form.addRow("Forward velocity (w1)", self.forward)
        form.addRow("Not falling (w2)", self.timestep)
        form.addRow("Power penalty (w3)", self.power)
        form.addRow("Vertical penalty (w4)", self.vertical)
        form.addRow("Lateral penalty (w5)", self.lateral)

    def as_dict(self):
        return {
            "forwardRewardWeight": self.forward.value(),
            "timestepRewardWeight": self.timestep.value(),
            "powerPenaltyWeight": self.power.value(),
            "verticalPenaltyWeight": self.vertical.value(),
            "lateralPenaltyWeight": self.lateral.value(),
        }


class JointControl(QGroupBox):
    """One stiffness + one damping value, applied to all three leg joints."""

    def __init__(self):
        super().__init__("Joint controller")
        form = QFormLayout(self)
        self.stiffness = dspin(0.0, 200.0, 80.0, 5.0, 1)
        self.damping = dspin(0.0, 10.0, 1.0, 0.1, 2)
        form.addRow("Stiffness K", self.stiffness)
        form.addRow("Damping B", self.damping)

    def as_dict(self):
        k, b = self.stiffness.value(), self.damping.value()
        return {
            "hipFrontalStiffness": k,
            "kneeStiffness": k,
            "ankleStiffness": k,
            "hipFrontalDamping": b,
            "kneeDamping": b,
            "ankleDamping": b,
        }


class GAPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.weights = RewardWeights(lateral_default=2.5)
        self.joints = JointControl()

        hyper = QGroupBox("GA settings")
        form = QFormLayout(hyper)
        self.n_points = ispin(2, 8, 4)
        self.gait_min = dspin(0.2, 3.0, 0.5, 0.1, 2)
        self.gait_max = dspin(0.2, 3.0, 2.0, 0.1, 2)
        self.max_gen = ispin(1, 500, 20)
        self.pop_size = ispin(10, 1000, 100, 10)
        self.fitness_limit = dspin(-10000, 0, -1000, 100, 0)
        form.addRow("Waypoints / joint", self.n_points)
        form.addRow("Gait period min (s)", self.gait_min)
        form.addRow("Gait period max (s)", self.gait_max)
        form.addRow("Max generations", self.max_gen)
        form.addRow("Population size", self.pop_size)
        form.addRow("Fitness limit", self.fitness_limit)

        for w in (self.weights, self.joints, hyper):
            v.addWidget(w)
        v.addStretch(1)

    def overrides(self):
        return {
            "reward": self.weights.as_dict(),
            "controller": self.joints.as_dict(),
            "nPoints": self.n_points.value(),
            "gaitPeriodMin": self.gait_min.value(),
            "gaitPeriodMax": self.gait_max.value(),
            "maxGenerations": self.max_gen.value(),
            "populationSize": self.pop_size.value(),
            "fitnessLimit": self.fitness_limit.value(),
        }


class RLPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.weights = RewardWeights(lateral_default=5.0)  # RL default is 5
        self.joints = JointControl()

        hyper = QGroupBox("DDPG settings")
        form = QFormLayout(hyper)
        self.max_ep = ispin(1, 10000, 4000, 100)
        self.stop_reward = dspin(0, 5000, 1000, 50, 0)
        self.save_reward = dspin(0, 5000, 500, 50, 0)
        form.addRow("Max episodes", self.max_ep)
        form.addRow("Stop at avg reward", self.stop_reward)
        form.addRow("Save agent above", self.save_reward)

        note = QLabel(
            "Live progress for RL shows in MATLAB's Episode Manager.\n"
            "Full training takes hours \u2014 use a pretrained agent for demos."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")

        agent_box = QGroupBox("Pretrained agent (used when Train is off)")
        av = QVBoxLayout(agent_box)
        self.agent_path = QLabel("Default: sm_humanoid_walker_saved_agent.mat")
        self.agent_path.setWordWrap(True)
        self.agent_path.setStyleSheet("color: gray; font-size: 11px;")
        browse = QPushButton("Browse for .mat\u2026")
        browse.clicked.connect(self._browse_agent)
        clear = QPushButton("Use default")
        clear.clicked.connect(self._clear_agent)
        row = QHBoxLayout()
        row.addWidget(browse)
        row.addWidget(clear)
        av.addWidget(self.agent_path)
        av.addLayout(row)
        self._agent_file = ""

        for w in (self.weights, self.joints, hyper, agent_box, note):
            v.addWidget(w)
        v.addStretch(1)

    def _browse_agent(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained agent .mat", "", "MAT files (*.mat)"
        )
        if path:
            self._agent_file = path
            self.agent_path.setText(path)

    def _clear_agent(self):
        self._agent_file = ""
        self.agent_path.setText("Default: sm_humanoid_walker_saved_agent.mat")

    def agent_file(self):
        return self._agent_file

    def overrides(self):
        return {
            "reward": self.weights.as_dict(),
            "controller": self.joints.as_dict(),
            "maxEpisodes": self.max_ep.value(),
            "stopReward": self.stop_reward.value(),
            "saveReward": self.save_reward.value(),
        }


class MetricCard(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        self.caption = QLabel(label)
        self.caption.setStyleSheet("color: gray; font-size: 11px;")
        self.value = QLabel("\u2014")
        self.value.setStyleSheet("font-size: 20px; font-weight: 600;")
        lay.addWidget(self.caption)
        lay.addWidget(self.value)

    def set(self, text):
        self.value.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Humanoid Walker \u2014 Control Panel")
        self.resize(1080, 720)
        self.controller = EngineController(PROJECT_DIR)
        self._build_ui()

        # GA progress poller.
        self.poll = QTimer(self)
        self.poll.setInterval(500)
        self.poll.timeout.connect(self._poll_progress)

        self._set_status("Starting MATLAB engine\u2026")
        self.run_btn.setEnabled(False)
        self.controller.start_engine(self._engine_ready, self._engine_failed)

    # -- UI ---------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left rail --------------------------------------------------------
        left = QVBoxLayout()
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems(["Genetic Algorithm", "Reinforcement Learning (DDPG)"])
        self.method.currentIndexChanged.connect(self._on_method_change)
        method_row.addWidget(self.method, 1)
        left.addLayout(method_row)

        self.stack = QStackedWidget()
        self.ga_panel = GAPanel()
        self.rl_panel = RLPanel()
        self.stack.addWidget(self.ga_panel)
        self.stack.addWidget(self.rl_panel)
        left.addWidget(self.stack, 1)

        self.train_check = QCheckBox("Train (uncheck = use pretrained)")
        self.parallel_check = QCheckBox("Use parallel (needs Parallel Computing Toolbox)")
        left.addWidget(self.train_check)
        left.addWidget(self.parallel_check)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        left.addLayout(btn_row)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(320)
        root.addWidget(left_wrap)

        # Right side -------------------------------------------------------
        right = QVBoxLayout()

        cards = QGridLayout()
        self.card_reward = MetricCard("Total reward")
        self.card_steps = MetricCard("Sim steps")
        self.card_best = MetricCard("Best fitness")
        self.card_status = MetricCard("Status")
        for i, c in enumerate(
            (self.card_reward, self.card_steps, self.card_best, self.card_status)
        ):
            cards.addWidget(c, 0, i)
        right.addLayout(cards)

        # Progress plot.
        self.fig_prog = Figure(figsize=(5, 2.2), tight_layout=True)
        self.ax_prog = self.fig_prog.add_subplot(111)
        self.ax_prog.set_title("Training progress")
        self.ax_prog.set_xlabel("Generation")
        self.ax_prog.set_ylabel("Best fitness")
        self.canvas_prog = FigureCanvas(self.fig_prog)
        right.addWidget(self.canvas_prog)

        # Reward-over-time plot.
        self.fig_rew = Figure(figsize=(5, 2.2), tight_layout=True)
        self.ax_rew = self.fig_rew.add_subplot(111)
        self.ax_rew.set_title("Reward over trial")
        self.ax_rew.set_xlabel("Time (s)")
        self.ax_rew.set_ylabel("Reward / step")
        self.canvas_rew = FigureCanvas(self.fig_rew)
        right.addWidget(self.canvas_rew)

        # TIER-2 SLOT: add height / velocity / joint-angle plots here once
        # those signals are logged in the .slx and returned by the bridge.

        self.status = QLabel("")
        self.status.setStyleSheet("color: gray;")
        right.addWidget(self.status)

        root.addLayout(right, 1)

    # -- engine callbacks -------------------------------------------------
    def _engine_ready(self, eng):
        self.controller.eng = eng
        self._set_status("Engine ready.")
        self.run_btn.setEnabled(True)

    def _engine_failed(self, msg):
        self._set_status(f"Engine failed: {msg}")
        self.card_status.set("No engine")

    # -- actions ----------------------------------------------------------
    def _on_method_change(self, idx):
        self.stack.setCurrentIndex(idx)

    def _current_method(self):
        return "ga" if self.method.currentIndex() == 0 else "rl"

    def _on_run(self):
        method = self._current_method()
        panel = self.ga_panel if method == "ga" else self.rl_panel
        overrides = panel.overrides()
        do_train = self.train_check.isChecked()
        use_parallel = self.parallel_check.isChecked()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(method == "ga" and do_train)
        self.card_status.set("Running")
        self._reset_plots()
        self._set_status(f"Running {method.upper()} \u2026")

        if method == "ga" and do_train:
            self.poll.start()  # stream the fitness curve

        agent_file = self.rl_panel.agent_file() if method == "rl" else ""

        self.controller.run_training(
            method,
            overrides,
            do_train,
            use_parallel,
            self._on_finished,
            self._on_failed,
            agent_file,
        )

    def _on_stop(self):
        self.controller.request_stop()
        self._set_status("Stop requested \u2014 finishing current generation\u2026")

    # -- results ----------------------------------------------------------
    def _on_finished(self, result):
        self.poll.stop()
        self._poll_progress()  # final read

        self.card_reward.set(f"{result['total_reward']:.1f}")
        self.card_steps.set(str(result["sim_steps"]))
        self.card_status.set("Done")

        if result["method"] == "ga":
            fval = result.get("fval", float("nan"))
            self.card_best.set(f"{fval:.1f}")
        else:
            ep = result.get("episode_reward")
            if ep is not None and ep.size:
                self._plot_episode_reward(ep)
                self.card_best.set(f"{ep.max():.1f}")

        self._plot_reward_trial(result["reward_time"], result["reward_data"])
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("Finished.")

    def _on_failed(self, msg):
        self.poll.stop()
        self.card_status.set("Error")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status(f"Error: {msg}")

    # -- plotting ---------------------------------------------------------
    def _reset_plots(self):
        for ax, title, xl, yl in (
            (self.ax_prog, "Training progress", "Generation", "Best fitness"),
            (self.ax_rew, "Reward over trial", "Time (s)", "Reward / step"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        self.canvas_prog.draw()
        self.canvas_rew.draw()

    def _poll_progress(self):
        gens, best = self.controller.read_progress()
        if not gens:
            return
        self.ax_prog.clear()
        self.ax_prog.set_title("Training progress")
        self.ax_prog.set_xlabel("Generation")
        self.ax_prog.set_ylabel("Best fitness")
        self.ax_prog.plot(gens, best, marker="o", ms=3)
        self.canvas_prog.draw()
        self.card_best.set(f"{best[-1]:.1f}")

    def _plot_episode_reward(self, ep):
        self.ax_prog.clear()
        self.ax_prog.set_title("Training progress")
        self.ax_prog.set_xlabel("Episode")
        self.ax_prog.set_ylabel("Episode reward")
        self.ax_prog.plot(range(1, ep.size + 1), ep, lw=1)
        self.canvas_prog.draw()

    def _plot_reward_trial(self, t, r):
        if t is None or not len(t):
            return
        self.ax_rew.clear()
        self.ax_rew.set_title("Reward over trial")
        self.ax_rew.set_xlabel("Time (s)")
        self.ax_rew.set_ylabel("Reward / step")
        self.ax_rew.plot(t, r, lw=1)
        self.canvas_rew.draw()

    # -- misc -------------------------------------------------------------
    def _set_status(self, text):
        self.status.setText(text)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
