# Experiments, November 2026 (summary)

Short record of experiments that did not become the main line, so they aren't repeated blindly. Their code and
long notes were removed on 2026-11-05 and are kept, not in git, in `out/removed_2026-11-05/` (patches, scripts,
tests, the original research notes).

## Wider joint range pilot (Nov 4)
`runs/v7/juan_range_pilot_20261104_*`: two 5M-step arms from the best walker with a wider action range, one widening
it gradually (to 0.6 rad), one at once (0.8 rad). Before training both started at about the reference's level
(6-8 failures in 80 trials, ~0.197 m/s at fast commands vs 0.194). The trained arms showed no clear winner.
Same conclusion as juan_v7_27_range: a wider range alone does not make it faster; the policy uses the extra
range at the knee for foot lift, not for longer strides.

## Speed from scratch, then wear refinement (Nov 4-5)
`runs/archive/v10/`: training a new walker from scratch for speed (later with a net centre-of-mass speed
measure and a signed forward-progress reward, so rocking doesn't count as speed), meant to be followed by a
servo-wear refinement. **Failed:** 44 checkpoints, none qualified. The best by fast-command speed reached
0.18-0.20 m/s but failed 44 of 80 trials, 28 of them simulated hardware-limit stops; the final one regressed.
The wear phase never ran. The fine-tuned v7 line (v7_28 / v7_29) is better on every count.
The optional reward terms it added (`tracking_window_s`, `forward_progress`, `loaded_reversals`) were removed
with it; the patch is `out/removed_2026-11-05/parked_reward_terms.patch`.

## What was kept from that work
- Physics fix: MuJoCo's derived mass constants are recomputed at every reset (`mj_setConst` in
  `BipedEnv._reset_to_home`), so the centre of mass is right on randomized robots. Runs before 2026-11-05 used
  slightly stale values; new trainings won't reproduce old ones exactly.
- Training options: `--seed`, `--deadline-monotonic`, `--min-std-rad`, `--runtime-guards` (NaN checks, a
  heartbeat file), `--eval-every`, `--eval-episodes`, `--video-every 0` (scripts/train.py,
  biped_rl/training/guards.py).
- `scripts/rescale_actions.py`: callable, refuses an existing destination, optional `--min-std-rad`.
- Pi safety: `scripts/deploy_pi.sh` checks the Pi's Git database before touching calibration or policies;
  `scripts/repair_pi_git.py` restores corrupt objects. The SD-card recovery is in [pi-recovery.md](pi-recovery.md).
