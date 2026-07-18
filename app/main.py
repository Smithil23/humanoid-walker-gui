"""
Humanoid Walker control GUI (PySide6).

Left rail  : pick method (GA / RL), tune the parameters that map to the actual
             MATLAB variables, and Run / Stop.
Right side : headline metrics, a live training-progress plot, the reward-over-
             trial plot, and a timestamped status log.

Run with:  python -m app.main   (from the humanoid_walker_gui/ folder)

The MATLAB engine and toolboxes are only needed to actually run training; the
window itself opens without them so you can develop the UI offline.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

os.environ.setdefault("QT_API", "PySide6")

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
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
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
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

# Matplotlib colours tuned to match the dark theme.
PLOT_BG = "#1b1e24"
PLOT_FG = "#c9d1d9"
PLOT_ACCENT = "#4f9cf9"
PLOT_GOOD = "#3fb950"

# Application-wide dark stylesheet.
STYLE = """
QWidget { background: #14171c; color: #c9d1d9; font-size: 13px; }
QGroupBox {
    border: 1px solid #2a2f38; border-radius: 8px; margin-top: 10px; padding: 8px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8b98a9; }
QLabel { background: transparent; }
QPushButton {
    background: #232833; border: 1px solid #313846; border-radius: 6px;
    padding: 6px 12px; color: #d7dee8;
}
QPushButton:hover { background: #2c333f; }
QPushButton:disabled { color: #5a6472; border-color: #23272f; }
QPushButton#run { background: #2563eb; border: none; color: white; font-weight: 600; }
QPushButton#run:hover { background: #2f6ff5; }
QPushButton#run:disabled { background: #24304a; color: #6b7686; }
QPushButton#stop { background: #3a2226; border: 1px solid #5a2a30; color: #f0a6ad; }
QPushButton#stop:disabled { color: #6b7686; border-color: #2a2f38; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #1b1f27; border: 1px solid #2f3644; border-radius: 5px; padding: 3px 6px;
    selection-background-color: #2563eb;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #3a4353; }
QComboBox QAbstractItemView { background: #1b1f27; selection-background-color: #2563eb; }
QCheckBox { spacing: 6px; }
QPlainTextEdit {
    background: #0f1216; border: 1px solid #2a2f38; border-radius: 6px;
    color: #a7b2c0; font-family: Consolas, "Courier New", monospace; font-size: 12px;
}
QProgressBar {
    background: #1b1f27; border: 1px solid #2a2f38; border-radius: 6px;
    text-align: center; height: 8px;
}
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
QFrame#card { background: #1b1f27; border: 1px solid #2a2f38; border-radius: 8px; }
QMenuBar { background: #14171c; }
QMenuBar::item:selected { background: #232833; }
QMenu { background: #1b1f27; border: 1px solid #2a2f38; }
QMenu::item:selected { background: #2563eb; }
"""


def dspin(lo, hi, val, step, decimals=3, tip=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    if tip:
        w.setToolTip(tip)
    return w


def ispin(lo, hi, val, step=1, tip=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    if tip:
        w.setToolTip(tip)
    return w


def style_axes(ax):
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_color("#2a2f38")
    ax.tick_params(colors=PLOT_FG, labelsize=8)
    ax.xaxis.label.set_color(PLOT_FG)
    ax.yaxis.label.set_color(PLOT_FG)
    ax.title.set_color(PLOT_FG)
    ax.grid(True, color="#242a33", linewidth=0.6)


class RewardWeights(QGroupBox):
    """The five reward weights, each with its own realistic range/step."""

    def __init__(self, lateral_default=2.5):
        super().__init__("Reward weights")
        form = QFormLayout(self)
        self.forward = dspin(0.0, 5.0, 1.0, 0.1, 2, "w1 — reward for forward velocity.")
        self.timestep = dspin(0.0, 5.0, 1.0, 0.1, 2, "w2 — reward per timestep survived.")
        self.power = dspin(0.0, 2e-3, 5e-4, 1e-4, 5, "w3 — penalty on power/energy use.")
        self.vertical = dspin(0.0, 50.0, 25.0, 1.0, 1, "w4 — penalty on vertical bounce.")
        self.lateral = dspin(0.0, 10.0, lateral_default, 0.1, 2, "w5 — penalty on sideways drift.")
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
        self.stiffness = dspin(
            0.0, 200.0, 80.0, 5.0, 1, "Spring stiffness K pulling each joint to its target angle."
        )
        self.damping = dspin(
            0.0, 10.0, 1.0, 0.1, 2, "Damping B resisting fast joint motion (the shock absorber)."
        )
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
        self.n_points = ispin(2, 8, 4, tip="Angular waypoints per joint in one gait cycle.")
        self.gait_min = dspin(0.2, 3.0, 0.5, 0.1, 2, "Lower bound on the gait period search.")
        self.gait_max = dspin(0.2, 3.0, 2.0, 0.1, 2, "Upper bound on the gait period search.")
        self.max_gen = ispin(1, 500, 20, tip="Number of GA generations (more = better, slower).")
        self.pop_size = ispin(10, 1000, 100, 10, "Candidates per generation.")
        self.fitness_limit = dspin(
            -10000, 0, -1000, 100, 0, "Early-stop target (more negative = better)."
        )
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

        algo_box = QGroupBox("Algorithm")
        af = QFormLayout(algo_box)
        self.model = QComboBox()
        self.model.addItems(["DDPG", "TD3", "SAC"])
        self.model.setToolTip(
            "DDPG: baseline. TD3: twin critics, usually more stable. "
            "SAC: stochastic actor, best exploration."
        )
        af.addRow("Model", self.model)

        self.weights = RewardWeights(lateral_default=5.0)
        self.joints = JointControl()

        hyper = QGroupBox("Training settings")
        form = QFormLayout(hyper)
        self.max_ep = ispin(1, 10000, 4000, 100, "Maximum training episodes.")
        self.stop_reward = dspin(0, 5000, 1000, 50, 0, "Stop when avg reward exceeds this.")
        self.save_reward = dspin(0, 5000, 500, 50, 0, "Save agents scoring above this.")
        form.addRow("Max episodes", self.max_ep)
        form.addRow("Stop at avg reward", self.stop_reward)
        form.addRow("Save agent above", self.save_reward)

        agent_box = QGroupBox("Pretrained agent (used when Train is off)")
        av = QVBoxLayout(agent_box)
        self.agent_path = QLabel("Default: sm_humanoid_walker_saved_agent.mat")
        self.agent_path.setWordWrap(True)
        self.agent_path.setStyleSheet("color: #8b98a9; font-size: 11px;")
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

        note = QLabel("Live training progress opens in MATLAB's Episode Manager.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b98a9;")

        for w in (algo_box, self.weights, self.joints, hyper, agent_box, note):
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

    def rl_model(self):
        return self.model.currentText().lower()

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
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        self.caption = QLabel(label)
        self.caption.setStyleSheet("color: #8b98a9; font-size: 11px;")
        self.value = QLabel("\u2014")
        self.value.setStyleSheet("font-size: 20px; font-weight: 600; color: #e6edf3;")
        lay.addWidget(self.caption)
        lay.addWidget(self.value)

    def set(self, text, color="#e6edf3"):
        self.value.setText(text)
        self.value.setStyleSheet(f"font-size: 20px; font-weight: 600; color: {color};")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Humanoid Walker \u2014 Control Panel")
        self.resize(1120, 760)
        self.controller = EngineController(PROJECT_DIR)
        self._last_result = None
        self._build_menu()
        self._build_ui()

        self.poll = QTimer(self)
        self.poll.setInterval(500)
        self.poll.timeout.connect(self._poll_progress)

        self.log("Starting MATLAB engine\u2026")
        self.run_btn.setEnabled(False)
        self.busy.setRange(0, 0)  # indeterminate while the engine starts
        self.controller.start_engine(self._engine_ready, self._engine_failed)

    # -- menu -------------------------------------------------------------
    def _build_menu(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        export = QAction("Export results (CSV)\u2026", self)
        export.triggered.connect(self._export_results)
        quit_a = QAction("Quit", self)
        quit_a.triggered.connect(self.close)
        file_menu.addAction(export)
        file_menu.addSeparator()
        file_menu.addAction(quit_a)

        help_menu = bar.addMenu("Help")
        about = QAction("About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

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
        self.method.addItems(["Genetic Algorithm", "Reinforcement Learning"])
        self.method.currentIndexChanged.connect(self._on_method_change)
        method_row.addWidget(self.method, 1)
        left.addLayout(method_row)

        self.stack = QStackedWidget()
        self.ga_panel = GAPanel()
        self.rl_panel = RLPanel()
        self.stack.addWidget(self.ga_panel)
        self.stack.addWidget(self.rl_panel)
        left.addWidget(self.stack, 1)

        self.train_check = QCheckBox("Train (unchecked = use pretrained)")
        self.train_check.setToolTip(
            "Off: play a ready-made result in seconds. On: train from scratch."
        )
        self.parallel_check = QCheckBox("Use parallel (needs Parallel Computing Toolbox)")
        self.parallel_check.setToolTip("Spread simulations across CPU cores. Leave off if unsure.")
        left.addWidget(self.train_check)
        left.addWidget(self.parallel_check)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        left.addLayout(btn_row)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(340)
        root.addWidget(left_wrap)

        # Right side -------------------------------------------------------
        right = QVBoxLayout()

        cards = QGridLayout()
        self.card_reward = MetricCard("Total reward")
        self.card_steps = MetricCard("Sim duration")
        self.card_best = MetricCard("Best fitness")
        self.card_status = MetricCard("Status")
        for i, c in enumerate(
            (self.card_reward, self.card_steps, self.card_best, self.card_status)
        ):
            cards.addWidget(c, 0, i)
        right.addLayout(cards)

        self.busy = QProgressBar()
        self.busy.setRange(0, 1)
        self.busy.setValue(0)
        right.addWidget(self.busy)

        self.fig_prog = Figure(figsize=(5, 2.0), facecolor=PLOT_BG, tight_layout=True)
        self.ax_prog = self.fig_prog.add_subplot(111)
        self.canvas_prog = FigureCanvas(self.fig_prog)
        right.addWidget(self.canvas_prog)

        self.fig_rew = Figure(figsize=(5, 2.0), facecolor=PLOT_BG, tight_layout=True)
        self.ax_rew = self.fig_rew.add_subplot(111)
        self.canvas_rew = FigureCanvas(self.fig_rew)
        right.addWidget(self.canvas_rew)

        self._reset_plots()

        log_box = QGroupBox("Log")
        lb = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        lb.addWidget(self.log_view)
        right.addWidget(log_box)

        root.addLayout(right, 1)

    # -- logging ----------------------------------------------------------
    def log(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {msg}")

    # -- engine callbacks -------------------------------------------------
    def _engine_ready(self, eng):
        self.controller.eng = eng
        self.log("Engine ready.")
        self.card_status.set("Ready", PLOT_GOOD)
        self.run_btn.setEnabled(True)
        self.busy.setRange(0, 1)
        self.busy.setValue(0)

    def _engine_failed(self, msg):
        self.log(f"Engine failed: {msg}")
        self.card_status.set("No engine", "#f0a6ad")
        self.busy.setRange(0, 1)
        self.busy.setValue(0)

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
        agent_file = self.rl_panel.agent_file() if method == "rl" else ""
        rl_model = self.rl_panel.rl_model() if method == "rl" else "ddpg"

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(method == "ga" and do_train)
        self.card_status.set("Running", "#e3b341")
        self.busy.setRange(0, 0)  # indeterminate
        self._reset_plots()

        label = rl_model.upper() if method == "rl" else "GA"
        mode = "training" if do_train else "pretrained"
        self.log(f"Running {label} ({mode})\u2026")

        if method == "ga" and do_train:
            self.poll.start()

        self.controller.run_training(
            method,
            overrides,
            do_train,
            use_parallel,
            self._on_finished,
            self._on_failed,
            agent_file,
            rl_model,
        )

    def _on_stop(self):
        self.controller.request_stop()
        self.log("Stop requested \u2014 finishing current generation\u2026")

    # -- results ----------------------------------------------------------
    def _on_finished(self, result):
        self.poll.stop()
        self._poll_progress()
        self._last_result = result

        steps = result["sim_steps"]
        secs = steps * 0.025  # Ts = 0.025 s
        self.card_reward.set(f"{result['total_reward']:.1f}")
        self.card_steps.set(f"{secs:.1f} s")

        # A short trial usually means the walker fell early.
        walked = secs > 10
        self.card_status.set("Walked" if walked else "Fell", PLOT_GOOD if walked else "#f0a6ad")

        if result["method"] == "ga":
            fval = result.get("fval", float("nan"))
            self.card_best.set(f"{fval:.1f}")
        else:
            ep = result.get("episode_reward")
            if ep is not None and ep.size:
                self._plot_episode_reward(ep)
                self.card_best.set(f"{ep.max():.1f}")

        self._plot_reward_trial(result["reward_time"], result["reward_data"])
        self.busy.setRange(0, 1)
        self.busy.setValue(0)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log(f"Finished. Total reward {result['total_reward']:.1f}, {secs:.1f} s.")

    def _on_failed(self, msg):
        self.poll.stop()
        self.card_status.set("Error", "#f0a6ad")
        self.busy.setRange(0, 1)
        self.busy.setValue(0)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log(f"Error: {msg}")

    # -- plotting ---------------------------------------------------------
    def _reset_plots(self):
        self.ax_prog.clear()
        self.ax_prog.set_title("Training progress")
        self.ax_prog.set_xlabel("Generation / episode")
        self.ax_prog.set_ylabel("Fitness / reward")
        style_axes(self.ax_prog)
        self.ax_rew.clear()
        self.ax_rew.set_title("Reward over trial")
        self.ax_rew.set_xlabel("Time (s)")
        self.ax_rew.set_ylabel("Reward / step")
        style_axes(self.ax_rew)
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
        style_axes(self.ax_prog)
        self.ax_prog.plot(gens, best, marker="o", ms=3, color=PLOT_ACCENT)
        self.canvas_prog.draw()
        self.card_best.set(f"{best[-1]:.1f}")

    def _plot_episode_reward(self, ep):
        self.ax_prog.clear()
        self.ax_prog.set_title("Training progress")
        self.ax_prog.set_xlabel("Episode")
        self.ax_prog.set_ylabel("Episode reward")
        style_axes(self.ax_prog)
        self.ax_prog.plot(range(1, ep.size + 1), ep, lw=1, color=PLOT_ACCENT)
        self.canvas_prog.draw()

    def _plot_reward_trial(self, t, r):
        if t is None or not len(t):
            return
        self.ax_rew.clear()
        self.ax_rew.set_title("Reward over trial")
        self.ax_rew.set_xlabel("Time (s)")
        self.ax_rew.set_ylabel("Reward / step")
        style_axes(self.ax_rew)
        self.ax_rew.plot(t, r, lw=1, color=PLOT_GOOD)
        self.canvas_rew.draw()

    # -- menu handlers ----------------------------------------------------
    def _export_results(self):
        if not self._last_result:
            QMessageBox.information(self, "Export", "Run something first \u2014 no results yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "results.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        r = self._last_result
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["metric", "value"])
                w.writerow(["method", r["method"]])
                w.writerow(["total_reward", r["total_reward"]])
                w.writerow(["sim_steps", r["sim_steps"]])
                if "fval" in r:
                    w.writerow(["fitness", r["fval"]])
                w.writerow([])
                w.writerow(["time_s", "reward_per_step"])
                for t, rr in zip(r["reward_time"], r["reward_data"]):
                    w.writerow([t, rr])
            self.log(f"Exported results to {path}")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _about(self):
        QMessageBox.about(
            self,
            "About",
            "Humanoid Walker Control GUI\n\n"
            "A PySide6 front-end for the MATLAB/Simulink humanoid walker, "
            "supporting genetic-algorithm and reinforcement-learning "
            "(DDPG / TD3 / SAC) training via the MATLAB Engine.",
        )


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
