import importlib.util

import pytest

from biped_rl.robots import PROJECT
from biped_rl.tasks.locomotion import EnvConfig

CHECKPOINT = PROJECT / "checkpoints" / "juan_v7_12b_pushes_best"


def load_script(relpath):
    """A script in scripts/ as a module, to call its functions."""
    path = PROJECT / "scripts" / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cfg():
    """The shipped policy's configuration, on nominal physics."""
    c = EnvConfig.load(CHECKPOINT)
    c.randomization.enabled = False
    return c
