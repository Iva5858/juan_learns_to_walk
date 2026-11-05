import importlib.util

import pytest

from biped_rl.robots import PROJECT
from biped_rl.tasks.locomotion import EnvConfig

CHECKPOINT = PROJECT / "checkpoints" / "juan_v7_12b_pushes_best"


@pytest.fixture(autouse=True, scope="session")
def silent_audio():
    """No sound from the test run: the fake robot's personality (dashboard and walker tests) would play its
    chirps on this computer's speakers. Voice keeps its state; it just has no player."""
    from robot import sound
    original = sound.Voice.__init__

    def muted(self, *a, **k):
        original(self, *a, **k)
        self.backend = None
    sound.Voice.__init__ = muted
    yield
    sound.Voice.__init__ = original


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
