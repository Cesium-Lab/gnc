"""Mission file naming and I/O shared by the Mars Courier notebooks.

Every stage writes `trajectories/<stage>-<mission_slug>.mission`, a NumPy NPZ
archive, e.g. `trajectories/departure-birthday_christmas_earlier.mission`.
Stages run in order: interplanetary -> departure -> arrival -> rpo; each loads
the archives of the stages before it.
"""

from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

MISSION_DIR = Path(__file__).parent / "trajectories"
STAGES = ("interplanetary", "departure", "arrival", "rpo")


def mission_slug(mission_name: str) -> str:
    """'birthday-christmas-earlier' -> 'birthday_christmas_earlier'."""
    return mission_name.replace("-", "_").replace(" ", "_")


def mission_path(stage: str, mission_name: str) -> Path:
    """Path of one stage's archive, e.g. trajectories/arrival-<slug>.mission."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return MISSION_DIR / f"{stage}-{mission_slug(mission_name)}.mission"


def save_mission(stage: str, mission_name: str, **fields) -> Path:
    """Write a stage archive; `stage` and `mission_name` are stored alongside `fields`."""
    path = mission_path(stage, mission_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(
            file, stage=stage, mission_name=mission_slug(mission_name), **fields
        )
    return path


def load_mission(stage: str, mission_name: str) -> NpzFile:
    """Open a stage archive (no pickles). Raises FileNotFoundError naming the notebook to run."""
    path = mission_path(stage, mission_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run {stage}.ipynb first")
    return np.load(path, allow_pickle=False)
