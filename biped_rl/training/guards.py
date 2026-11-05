"""Optional wall-clock stop, physical exploration floor and progress heartbeat."""
import json
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class RuntimeGuards(BaseCallback):
    def __init__(self, scale, deadline=None, min_std_rad=None, progress=None):
        super().__init__()
        self.scale, self.deadline, self.min_std_rad, self.progress = np.asarray(scale), deadline, min_std_rad, progress
        self.last_write = -np.inf
        self.reason = "training"

    def floor_noise(self):
        if self.min_std_rad is not None:
            with torch.no_grad():
                std = self.model.policy.log_std
                floor = torch.as_tensor(np.log(self.min_std_rad / self.scale), dtype=std.dtype, device=std.device)
                std.copy_(torch.maximum(std, floor))

    def _on_rollout_start(self):
        if not all(torch.isfinite(p).all() for p in self.model.policy.parameters()):
            raise ValueError("nonfinite policy parameter")
        self.floor_noise()

    def heartbeat(self, force=False):
        if self.progress is not None and (force or time.monotonic() - self.last_write >= 30):
            path = self.progress
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(state=self.reason, steps=self.num_timesteps,
                           updated=datetime.now().astimezone().isoformat(timespec="seconds"))) + "\n")
            tmp.replace(path)
            self.last_write = time.monotonic()

    def _on_step(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.reason = "deadline_reached"
            self.heartbeat(True)
            return False
        self.heartbeat()
        return True

    def _on_training_end(self):
        self.floor_noise()
        if self.reason == "training":
            self.reason = "learning_finished"
        self.heartbeat(True)
