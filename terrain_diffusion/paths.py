import os
from pathlib import Path

# The project root is two levels up from this file (terrain_diffusion/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default data directory is 'data' in the project root.
# Can be overridden by TERRAIN_DATA_DIR environment variable.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.getenv("TERRAIN_DATA_DIR", DEFAULT_DATA_DIR))

# Default checkpoint directory is 'checkpoints' in the project root.
# Can be overridden by TERRAIN_CHECKPOINT_DIR environment variable.
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR = Path(os.getenv("TERRAIN_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))

def get_data_path(rel_path: str) -> str:
    """
    Get the absolute path for a data file, given its path relative to the data directory.
    If the provided path is already absolute, it is returned as is.
    """
    if not rel_path:
        return str(DATA_DIR)
    
    path = Path(rel_path)
    if path.is_absolute():
        return str(path)
    
    # If it starts with 'data/', strip it because DATA_DIR already points to 'data'
    # This helps with refactoring existing hardcoded 'data/...' paths.
    parts = path.parts
    if parts and parts[0] == "data":
        return str(DATA_DIR.joinpath(*parts[1:]))
        
    return str(DATA_DIR / rel_path)

def get_project_path(rel_path: str) -> str:
    """
    Get the absolute path for a project file, given its path relative to the project root.
    """
    if not rel_path:
        return str(PROJECT_ROOT)
        
    path = Path(rel_path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / rel_path)

def get_checkpoint_path(rel_path: str) -> str:
    """
    Get the absolute path for a checkpoint, given its path relative to the checkpoint directory.
    """
    if not rel_path:
        return str(CHECKPOINT_DIR)
        
    path = Path(rel_path)
    if path.is_absolute():
        return str(path)
    
    # If it starts with 'checkpoints/', strip it
    parts = path.parts
    if parts and parts[0] == "checkpoints":
        return str(CHECKPOINT_DIR.joinpath(*parts[1:]))
        
    return str(CHECKPOINT_DIR / rel_path)
