# Juan learns to walk

Reinforcement learning for **Juan**, a 43 cm, 8-servo biped. The policy is trained in MuJoCo with PPO and runs on the real robot's Raspberry Pi, steered from a web dashboard.

**Status (October 2026):** Juan walks on the real robot at up to 0.2 m/s, turns, side-steps and survives light shoves, with the policy in `robot/policies/juan_v7_12b_pushes_best.npz`. It weaves left and right while walking straight. The fine-tune for that (v7_13) trains on the servo model fitted from the robot's own walking logs.

| | |
|---|---|
| Size, weight | 43 cm tall, ~2 kg |
| Legs | 4 joints each: hip roll, hip pitch, knee, ankle |
| Servos | 8 × Hiwonder HTD-35H serial bus servos (3.4 Nm, 5.8 rad/s at 11.1 V), one BusLinker USB board per leg |
| Computer | Raspberry Pi 4, policy at 50 Hz |
| Sensors | LSM9DS1 IMU (accelerometer + gyro), servo angle readback, Pi camera |
| Power | 3S LiPo for the servos, a separate pack for the Pi |

---

## Quick start

```bash
# setup (uv: https://docs.astral.sh/uv/). Installs the exact versions in uv.lock, the ones every result here came from
uv sync
source .venv/bin/activate
pytest                                         # ~5 s: config, simulator, policy export, robot joint table, gestures

# drive the trained policy with the arrow keys (macOS needs mjpython for the viewer)
mjpython scripts/play.py                       # boot sequence, then the policy; --robust / --pushes for harder physics

# how well does it walk?
python scripts/evaluate.py -n 3                # speed, gait, impacts, smoothness; video in out/eval/
python scripts/analysis/straight.py checkpoints/juan_v7_12b_pushes_best    # drift, turning and weave
python scripts/analysis/falls.py checkpoints/juan_v7_12b_pushes_best       # falls with and without shoves

# fine-tune it (e.g. a reward change); ~50 min per 30M steps on an 18-core Mac
python scripts/train.py --run-name my_run --stage pushes --resume checkpoints/juan_v7_12b_pushes_best \
    --timesteps 30000000 --lr 1e-4 --reward heading=2.0
tensorboard --logdir runs                      # charts (see docs/training.md)

# put it on the robot
python scripts/export_policy.py runs/my_run    # -> robot/policies/my_run.npz, checked step by step against the sim
```

`checkpoints/` holds the training state (network, normalization, curricula, config) of the policy on the robot, so you can fine-tune from it. Training outputs go to `runs/<name>/` (not in git). Run everything from the project folder, and keep the project out of iCloud-synced folders: iCloud turns checkpoints into placeholders that hang training.

---

## How it works

```
Onshape CAD ─onshape-to-robot─▶ juan/robot.xml ─scripts/model/postprocess_onshape.py─▶ juan/rl_scene.xml
                                                                                            │
biped_rl/robots/juan.py ───▶ biped_rl/tasks/locomotion/ ◀───────────────────────────────────┘
(joints, servo numbers)      env.py       simulation loop, servo model
                             mdp/         commands, observations, rewards, terminations,
                                          events (randomization, pushes), curricula
                             env_cfg.py   every setting
                                   │
                                   ▼
            scripts/train.py + biped_rl/training/ (PPO, stages, TensorBoard)
                                   │
                                   ▼
          runs/<name>/ ──▶ scripts/evaluate.py, play.py, analysis/
                       ──▶ scripts/export_policy.py ──▶ robot/policies/*.npz ──▶ robot/ (on the Pi)
```

The layout follows Berkeley Humanoid Lite [[1]](#references): a robot description, a task made of small term modules, a training package and thin scripts. The training recipe follows Li et al. [[2]](#references).

See [docs/training.md](docs/training.md) for the servo model, observations, commands, reward terms, randomization, training stages, TensorBoard, the analysis tools, updating the model from Onshape and troubleshooting.

---

## The real robot

Everything in [robot/](robot/) runs on the Raspberry Pi and needs only `pyserial`, `smbus2` and `numpy`, not the training stack: `python3 -m robot.dashboard` serves a web page to calibrate, monitor and drive Juan. Details in [docs/robot.md](docs/robot.md); setup and first power-on, step by step, in [docs/bringup.md](docs/bringup.md).

---

## Project layout

| Path | What |
|---|---|
| [biped_rl/robots/juan.py](biped_rl/robots/juan.py) | Juan: model path, feet, joint conventions, mirror pairs, servo numbers (estimated and measured) |
| [biped_rl/tasks/locomotion/](biped_rl/tasks/locomotion/) | The walking task: [env.py](biped_rl/tasks/locomotion/env.py) (simulation loop, servo model), [env_cfg.py](biped_rl/tasks/locomotion/env_cfg.py) (every setting), [mdp/](biped_rl/tasks/locomotion/mdp/) (commands, observations, rewards, terminations, events, curricula) |
| [biped_rl/training/](biped_rl/training/) | PPO hyperparameters, training stages, TensorBoard dashboard and clips |
| [biped_rl/policy.py](biped_rl/policy.py), [symmetry.py](biped_rl/symmetry.py), [motions.py](biped_rl/motions.py), [runs.py](biped_rl/runs.py) | Asymmetric mirror-symmetric actor-critic; mirror maps; the boot sequence; loading a trained run |
| [scripts/](scripts/) | Training, evaluation, export, checks; [analysis/](scripts/analysis/); [model/](scripts/model/) (CAD export → RL model) |
| [checkpoints/](checkpoints/) | Training state of the policy on the robot, to fine-tune from |
| [robot/](robot/) | Runs on the Pi; [policies/](robot/policies/) are the exported policies |
| [juan/](juan/) | The model: Onshape export (`robot.xml`, `scene.xml`, meshes in `assets/`) and the RL-ready `rl_robot.xml` / `rl_scene.xml` |
| [tests/](tests/) | `pytest`: run before every commit (CI runs them on every push) |
| [docs/](docs/) | [training.md](docs/training.md), [robot.md](docs/robot.md), [bringup.md](docs/bringup.md) |

---

## License

- **Code** (everything except the two items below): [Apache License 2.0](LICENSE). Use it in your own projects, commercial or not; keep the [NOTICE](NOTICE) file with any copy you distribute, so the credit stays.
- **Robot design** (`juan/`: the model and meshes) and **trained policies** (`checkpoints/`, `robot/policies/`): [CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt). Use and adapt them for non-commercial purposes with credit; ask for commercial use.

Everything is provided as is, without warranty and without support. If you use this work, please cite it ([CITATION.cff](CITATION.cff), or "Cite this repository" on GitHub).

---

## References

1. Y. Chi, Q. Liao, J. Long, X. Huang, S. Shao, B. Nikolic, Z. Li and K. Sreenath, "Demonstrating Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot," arXiv:2504.17249, 2025. [Paper](https://arxiv.org/abs/2504.17249) · [Code](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite). The code structure and the varied starting states.
2. Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth and K. Sreenath, "Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control," *The International Journal of Robotics Research*, vol. 44, no. 5, pp. 840–888, 2025. [arXiv:2401.16889](https://arxiv.org/abs/2401.16889). The staged training and the variety of commands as a source of robustness.
