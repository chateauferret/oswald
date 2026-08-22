import os
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.getenv("OSWALD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".oswald-root").exists():
            return candidate

    # Fallback for direct source-tree usage when the sentinel file is unavailable.
    # This assumes oswald/paths.py lives one directory below the project root.
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(
    os.getenv("OSWALD_DATA_DIR")
    or os.getenv("TERRAIN_DATA_DIR")
    or DEFAULT_DATA_DIR
)

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR = Path(
    os.getenv("OSWALD_CHECKPOINT_DIR")
    or os.getenv("TERRAIN_CHECKPOINT_DIR")
    or DEFAULT_CHECKPOINT_DIR
)


def get_data_path(rel_path: str) -> str:
    if not rel_path:
        return str(DATA_DIR)

    path = Path(rel_path)
    if path.is_absolute():
        return str(path)

    parts = path.parts
    if parts and parts[0] == "data":
        return str(DATA_DIR.joinpath(*parts[1:]))

    return str(DATA_DIR / rel_path)


def get_project_path(rel_path: str) -> str:
    if not rel_path:
        return str(PROJECT_ROOT)

    path = Path(rel_path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / rel_path)


def get_checkpoint_path(rel_path: str) -> str:
    if not rel_path:
        return str(CHECKPOINT_DIR)

    path = Path(rel_path)
    if path.is_absolute():
        return str(path)

    parts = path.parts
    if parts and parts[0] == "checkpoints":
        return str(CHECKPOINT_DIR.joinpath(*parts[1:]))

    return str(CHECKPOINT_DIR / rel_path)
