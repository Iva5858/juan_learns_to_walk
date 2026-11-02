# Runs: where training output goes

`runs/` is not in git. Everything in it follows one layout so any run can be found, traced back to where it
came from, and compared.

```
runs/
  INDEX.md                 every run, by generation (generated: python scripts/runs_index.py)
  v7/                      a generation: one line of training from scratch, and every fine-tune of it
    juan_v7_13_straight/   a run
    juan_v7_13_straight_best/   its best checkpoint, as a run of its own (made by finetune.sh and friends)
  v8/  v9/  ...
  v9/media/                videos that belong to the generation, not to one run
  hop/  early/             other tasks / the walks before the juan_vN names
  archive/v8/  archive/v9/ failed or abandoned runs, kept for reference
  logs/v7/  logs/v8/ ...   training logs and reports, by generation
  test/                    smoke tests and quick experiments: delete freely
  MOVES.txt                what the 2026-11-01 reorganization moved where
```

## Names

- **A run name is `<generation>/<name>`**: `python scripts/train.py --run-name v7/juan_v7_21_speed ...`.
  train.py refuses a bare name. The shell scripts take the same form
  (`./scripts/finetune.sh v7/juan_v7_21_speed`), and write their reports to `runs/logs/<generation>/`.
- **`juan_v<generation>_<number>_<what it is for>`**, e.g. `juan_v7_21_speed`. The number goes up by one
  for each new fine-tune in the generation, whichever run it starts from (the parent is recorded, not
  encoded in the name). Letters (`_15b`, `_15c`) are variants of the same recipe tried side by side.
- **Suffixes:** `_best`: the best checkpoint of the run in front of it. `_at52M`: a checkpoint of it at
  52M steps. `_test`: a short trial of a recipe.
- **A new generation** (v10, ...) only when training starts from scratch or from a new pre-trained network.
- The exported policy is `robot/policies/<name>.npz` and the robot's logs carry that name, so **never
  rename a run once it has been on the robot**. Move it between folders freely.

## What each run holds

- `config.json`: every setting it trained with; `curriculum.json`: where its curricula ended.
- `final_model.zip` + `vecnormalize.pkl`, and `best_model.zip` + `best_vecnormalize.pkl`.
- `run.json`: written by train.py. It holds the parent (`--resume`), stage, steps asked for, total steps (counting the runs it resumed), the full command line, the
  commit, and the start and finish times. For runs from before 2026-11-01 it was backfilled from the logs where
  the parent was known (`"backfilled": true`).
- `note.txt` (optional): one line on what it is good or bad for. Set it with
  `python scripts/runs_index.py --note v7/juan_v7_21_speed "..."`.

## Housekeeping

- After a training session: `python scripts/runs_index.py` (and add notes for what you learned).
- A run that failed: `mv runs/v8/juan_v8x runs/archive/v8/`. Delete archived runs only when sure.
- Safe policies to fine-tune from also live in `checkpoints/` (in git).
