"""Robot descriptions, and where to find their MuJoCo models."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def resolve_model_path(path):
    """Find a robot MJCF even if the project moved (new computer, new user name, out of iCloud).

    Relative paths are relative to the project folder. An absolute path that no longer exists
    is matched by its tail: /old/place/Juan_reinforcement_learning/juan/rl_scene.xml ->
    <project>/juan/rl_scene.xml."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT / p
    if p.exists():
        return str(p)
    parts = Path(path).parts
    for i in range(1, len(parts)):
        candidate = PROJECT.joinpath(*parts[i:])
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"robot model not found: {path} (also looked inside {PROJECT})")


def portable_model_path(path):
    """Path to save in a run's config.json: relative to the project when it is inside it."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT))
    except ValueError:
        return str(p)
