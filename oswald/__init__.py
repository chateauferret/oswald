"""Oswald runtime helpers."""
import sys
from oswald.paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oswald.globe_viewer import GlobeViewer, globe_viewer
from oswald.diffusion_server import DiffusionServer, diffusion_server, create_app
