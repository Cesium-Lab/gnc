"""Mission file naming and I/O shared by the Mars Courier notebooks.

Every stage writes `trajectories/<stage>-<mission_slug>.mission`, a NumPy NPZ
archive, e.g. `trajectories/departure-birthday_christmas_earlier.mission`.
Stages run in order: interplanetary -> departure -> arrival -> rpo; each loads
the archives of the stages before it.

`run_pipeline()` (or `python mission_io.py [--interplanetary]`) re-executes the
notebooks headless in that order, regenerating every archive and figure. Notebook
outputs are not saved; open a notebook and run it to see its plots inline.
"""

import json
import subprocess
import sys
import time
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


################################################################################
#               Headless pipeline
################################################################################

PROJECT_DIR = Path(__file__).parent
NOTEBOOK_ORDER = (
    "interplanetary",
    "departure",
    "ground_segment",
    "arrival",
    "rpo",
    "summary",
)
_PRELUDE = f"""
import sys; sys.path.insert(0, {str(PROJECT_DIR)!r})
import matplotlib; matplotlib.use("Agg")
import plotly.graph_objects as _go; _go.Figure.show = lambda self, *a, **k: None
"""


FIGURE_DIR = PROJECT_DIR / "figures"


def export_fig(
    fig, stage: str, name: str, width: int = 1600, height: int | None = None
):
    """Save a plotly or matplotlib figure to figures/<stage>/<name>.png for slides."""
    out = FIGURE_DIR / stage / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_image"):  # plotly
        fig.write_image(
            out, width=width, height=height or fig.layout.height or 800, scale=2
        )
    else:  # matplotlib
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out.relative_to(PROJECT_DIR)}")
    return out


def notebook_script(name: str) -> str:
    """A notebook's code cells as one script: IPython magics dropped, figures not shown."""
    nb = json.loads((PROJECT_DIR / f"{name}.ipynb").read_text())
    cells = [
        "\n".join(
            line
            for line in "".join(c["source"]).split("\n")
            if not line.lstrip().startswith(("%", "!"))
        )
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]
    return _PRELUDE + "\n\n".join(cells)


def run_notebook(name: str, quiet: bool = True) -> float:
    """Execute one notebook headless in a fresh process (cwd = project dir); returns seconds."""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", notebook_script(name)],
        cwd=PROJECT_DIR,
        check=False,  # failures are reported below with the stderr tail
        capture_output=quiet,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-3000:] if quiet else ""
        raise RuntimeError(f"{name}.ipynb failed (exit {proc.returncode})\n{tail}")
    return time.time() - t0


def run_pipeline(
    include_interplanetary: bool = False,
    include_summary: bool = True,
    quiet: bool = True,
) -> None:
    """Re-run the stage notebooks in order, stopping at the first failure.

    Interplanetary is skipped by default: it is slow and only changes when the
    trajectory choice does. Pass include_summary=False when calling from summary.ipynb.
    """
    for name in NOTEBOOK_ORDER:
        if name == "interplanetary" and not include_interplanetary:
            continue
        if name == "summary" and not include_summary:
            continue
        print(f"running {name}.ipynb ...", end=" ", flush=True)
        print(f"done in {run_notebook(name, quiet=quiet):.1f} s", flush=True)


if __name__ == "__main__":
    run_pipeline(include_interplanetary="--interplanetary" in sys.argv[1:])
