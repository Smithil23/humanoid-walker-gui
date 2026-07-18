# Humanoid Walker — Control GUI

![CI](https://github.com/Smithil23/humanoid-walker-gui/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **Python/PySide6 desktop application** that drives MathWorks' *Train Humanoid
Walker* Simscape Multibody example. Pick reward and controller parameters, launch
either **genetic-algorithm** or **reinforcement-learning (DDPG)** training through
the MATLAB Engine, watch progress live, and see the results plotted back in the
GUI — with the 3-D gait rendered in Simulink's Mechanics Explorer.

> Built as a software-engineering wrapper around a proprietary robotics
> simulation: a clean Python control/visualisation layer, a thin MATLAB bridge,
> background-threaded execution, and a full CI/CD pipeline. The humanoid model
> itself is MathWorks' example (see [Attribution](#attribution--licensing)); the
> value added here is the surrounding application and engineering.

<!-- Add screenshots to docs/ and they will render here -->
<p align="center">
  <img src="docs/gui.png" alt="Control GUI" width="70%"><br>
  <em>Parameter panel, live training curve, and reward trace.</em>
</p>

<p align="center">
  <img src="docs/walker.png" alt="Humanoid walker in Mechanics Explorer" width="70%"><br>
  <em>The trained walker rendered in Simulink Mechanics Explorer.</em>
</p>

## What it does

- **One control panel for two very different learning methods.** Switch between a
  genetic algorithm (open-loop gait optimisation) and a DDPG reinforcement-learning
  agent (closed-loop feedback policy), each with the parameters that actually
  matter surfaced as controls.
- **Live training feedback.** The GA's best-fitness-per-generation streams into the
  GUI in real time; the RL episode-reward curve is plotted when training returns.
- **Instant playback of pretrained results.** Uncheck *Train* to load a ready-made
  agent/solution and simulate it in seconds — including your own saved `.mat`
  agents via a file picker.
- **Results back in Python.** Reward-over-trial trace, headline metrics (total
  reward, sim steps, best fitness, fall status), all derived from the MATLAB run.

## Architecture

```
Python GUI (PySide6)                MATLAB Engine                 Simulink / Simscape
──────────────────────   JSON args  ─────────────  set params    ───────────────────
 parameter panels    ─────────────▶ gui_run_ga.m  ────────────▶  sm_humanoid_walker_ga
 Run / Stop                         gui_run_rl.m                  sm_humanoid_walker_rl
 live plots          ◀───────────── numeric arrays ◀──────────── logged Reward signal
```

The GUI never pokes MATLAB workspace variables directly. Instead it passes a JSON
blob of overrides to one of two **bridge functions** (`matlab_bridge/*.m`), which
apply them onto the real parameter struct, run training, and return plain numeric
arrays. This keeps the MathWorks example files untouched and gives Python a single
clean call per path. The MATLAB engine runs on a background thread so the UI stays
responsive; the GA streams progress through a `ga` `OutputFcn` that writes a CSV
the GUI polls.

## The two methods

| | Genetic Algorithm | Reinforcement Learning (DDPG) |
|---|---|---|
| Controller | Open-loop repeating gait (CPG-style) | Closed-loop feedback policy |
| Optimises | 13 vars: 12 joint waypoints + gait period | Actor/critic network weights |
| Toolbox | Global Optimization | Deep Learning + Reinforcement Learning |
| Live view | Streams to the GUI | MATLAB Episode Manager |
| Training time | Minutes | Hours |

## Results & analysis

Both methods optimise the same reward
(`r = w1·v_y + w2·Ts − w3·p − w4·Δz − w5·Δx`, summed over the trial), but they
behave very differently — which is the point of shipping both.

**Genetic algorithm.** The pretrained GA solution walks stably for the full
trial (~2100 steps, ~52 s, total reward ≈ 1520) with a clean periodic reward
signature — one oscillation per stride. A deliberately short 5-generation run
(population 20) is enough to *watch* fitness improve (−41 → −65) but produces a
walker that stumbles after a few seconds — expected, and useful for demonstrating
the streaming curve without a long run.

**DDPG.** A 4000-episode agent trained here reached a best single-episode reward
of ~390 but **did not converge to walking**: the final policy settles into a
crouched, near-stationary posture that avoids catastrophic falls (small survival
reward) rather than stepping forward — a textbook local optimum, visible as a
reward curve that hovers near zero for most of training and only spikes late. This
is an honest and instructive outcome: DDPG here learned *postural stability*, not
*locomotion*, and would need reward reshaping (stronger forward-velocity weight,
weaker survival term) and/or longer training with better exploration to escape it.

The contrast is the takeaway: for this model the GA's structured gait search finds
walking more readily than model-free RL, which is exactly why the original example
provides both.

## Setup

Requires a local MATLAB **R2025b** install (full install, not Runtime) with the
relevant toolboxes, and a compatible Python (3.9–3.12).

```bash
pip install -r requirements.txt
pip install matlabengine==25.2.*     # must match your MATLAB release
```

Point the app at your copy of the example (obtained via `openExample`, see below):

```bash
# Windows
set HW_PROJECT_DIR=C:\path\to\TrainHumanoidWalkerExample
# or edit PROJECT_DIR at the top of app/main.py
```

Run:

```bash
python -m app.main
```

The window opens immediately and starts the MATLAB engine in the background; *Run*
enables once it reports **Engine ready**.

**Toolboxes:** GA path needs Global Optimization Toolbox (+ Parallel Computing
Toolbox only if *Use parallel* is ticked). RL path needs Deep Learning Toolbox +
Reinforcement Learning Toolbox. All require Simscape Multibody.

## Usage

1. Choose **Genetic Algorithm** or **Reinforcement Learning (DDPG)**.
2. Adjust reward weights / controller / method settings.
3. Leave **Train** unchecked to play a pretrained result in seconds, or check it to
   train from scratch. For RL you can browse to your own trained-agent `.mat`.
4. **Run**. GA fitness streams into the top plot; the reward trace and metrics fill
   in when the run completes, and the gait renders in Mechanics Explorer.

## Development

```bash
pip install -e ".[dev]"      # ruff + pytest
pre-commit install           # optional: format/lint on commit
ruff check . && ruff format --check .
pytest -q                    # pure-Python tests, no MATLAB needed
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR across Python 3.9–3.12: ruff lint,
ruff format check, byte-compile, and pytest. CI deliberately does **not** run MATLAB
— hosted runners have no MATLAB licence — so the tests cover the MATLAB-independent
seam (array conversion, progress-file parsing, stop-flag handling). This is the
standard pattern for a project with a heavy licensed dependency: test the code you
own, don't pretend to test the engine.

## Project layout

```
app/            PySide6 GUI + MATLAB-engine threading
matlab_bridge/  thin .m wrappers (gui_run_ga, gui_run_rl)
tests/          pure-Python tests
.github/        CI workflow
```

## Attribution & licensing

The original code in this repository (the PySide6 GUI in `app/` and the bridge
scripts in `matlab_bridge/`) is released under the **MIT License** (see `LICENSE`).

The humanoid model, Simulink models, CAD geometry, and example scripts
(`sm_humanoid_walker_*`, `.slx`, `.stp`, `.mat`, `.mlx`) are **Copyright The
MathWorks, Inc.** and are **not** distributed here. Obtain them from your own
licensed MATLAB installation:

```matlab
openExample("sm/ImportedURDFExample")
```

Then point `HW_PROJECT_DIR` at that folder.
