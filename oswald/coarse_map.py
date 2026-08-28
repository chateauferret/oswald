"""
Class for loading and manipulating coarse maps. Coarse map is a raster or stack of rasters containing
elevation, precipitation and temperature data for the whole world in equirectangular format, having
width = 2 * height and data stored in .tif format on disk and loaded into memory as numpy arrays.

Accepts coarse map data as 2D rasters. For each channel accept a string denoting the channel together with one of the following:
    - numpy arrays (heightmap / elevation)
    - paths to a .npy or .npz file (or directory of .npy files)
    - paths to a .tif / .tiff or .npz file
    
Returns a 3D numpy array compatible with terrain_diffusion where channels correspond to 0..4

Usage (constructs model from one or more channels, loads model once to GPU, and generates multiple maps):
  from terrain_diffusion.inference.coarse_map import CoarseMap
  mapper = cm.CoarseMap (size=64, extent=3000)
  mapper.add_channel (coarse_map = world_coarse_map, channel = "heightmap")
  cond_map = mapper.get_cond_map (lat = 45.0, lon = 45.0)

"""
import numpy as np
from pathlib import Path
import rasterio
from oswald.utils import extract_channels, CHANNEL_INFO, CHANNEL_ALIASES, sample_equirectangular

from PIL import Image



class CoarseMap:

    def __init__(self, size: int = 64, extent: float | int = 3000):
        self.size = int(size)
        self.extent = float(extent)
        self.coarse_map: dict[int, np.ndarray] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_channel(self, coarse_map=None, channel: str | int = "heightmap", **kwargs):
        """Add a channel to the coarse map stack.

        Accepts:
            - coarse_map: 2D numpy array, PIL Image, Path, or file path (.tif, .npy, .npz, etc.)
            - channel: channel name or index (0..4)
        """
        # Handle kwargs or reversed argument names if needed
        data = coarse_map
        if data is None:
            if "file_path" in kwargs:
                data = kwargs["file_path"]
            elif "array" in kwargs:
                data = kwargs["array"]
            elif isinstance(channel, str) and channel in kwargs:
                data = kwargs[channel]

        if isinstance(data, Image.Image):
            data = np.array(data)

        if isinstance(data, (str, Path)):
            return self.load_coarse_map_from_file(data, channel=channel)
        elif isinstance(data, np.ndarray):
            return self.load_course_map_from_array(data, channel=channel)
        elif data is not None:
            arr = np.asarray(data)
            return self.load_course_map_from_array(arr, channel=channel)
        else:
            raise ValueError("No coarse map data provided to add_channel.")

    # Given a file_path to a coarse map stack (.tif, .npy, .npz, etc.), load it into memory and store it as a numpy array
    # in the coarse_map dictionary
    def load_coarse_map_stack_from_file(self, file_path):
        path = Path(file_path)
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(file_path) as src:
                    data = src.read()  # (C, H, W)
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                self.load_coarse_map_stack_from_array(data)
            except Exception:
                img = Image.open(file_path)
                data = np.array(img)
                self.load_coarse_map_stack_from_array(data)
        else:
            channels = extract_channels(file_path)
            self.coarse_map.update(channels)
        return self

    def load_coarse_map_stack_from_array(self, array=None, **kwargs):
        channels = extract_channels(array, **kwargs)
        self.coarse_map.update(channels)
        return self

    def load_course_map_from_array(self, array, channel: str | int = "heightmap"):
        if isinstance(channel, str):
            ch_lower = channel.lower()
            if ch_lower in CHANNEL_ALIASES:
                ch_idx = CHANNEL_ALIASES[ch_lower]
            else:
                try:
                    ch_idx = int(channel)
                except ValueError:
                    raise ValueError(f"Unknown channel '{channel}'. Valid names: {list(CHANNEL_ALIASES.keys())}")
        elif isinstance(channel, int):
            ch_idx = channel
        else:
            raise ValueError(f"Invalid channel type: {type(channel)}")

        arr = np.asarray(array)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[2] == 1:
                arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Channel {channel} array must be 2D, got shape {arr.shape}")

        self.coarse_map[ch_idx] = arr
        return self

    load_coarse_map_from_array = load_course_map_from_array

    def load_coarse_map_from_file(self, file_path: str | Path, channel: str | int = "heightmap"):
        # Handle case where arguments are passed as (channel, file_path)
        actual_path = file_path
        actual_channel = channel
        if isinstance(file_path, str) and (file_path.lower() in CHANNEL_ALIASES or str(channel).lower().endswith(('.tif', '.tiff', '.npy', '.npz', '.png'))):
            if not Path(file_path).exists() or file_path.lower() in CHANNEL_ALIASES:
                actual_path = channel  # type: ignore
                actual_channel = file_path

        path = Path(str(actual_path))
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(path) as src:
                    arr = src.read(1)
            except Exception:
                img = Image.open(path)
                arr = np.array(img)
        elif path.suffix.lower() == ".npy":
            arr = np.load(path)
        elif path.suffix.lower() == ".npz":
            loaded = np.load(path)
            if isinstance(actual_channel, str) and actual_channel in loaded:
                arr = loaded[actual_channel]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError(f"Ambiguous channel in .npz: available keys {loaded.files}")
        else:
            img = Image.open(path)
            arr = np.array(img)

        return self.load_course_map_from_array(arr, channel=actual_channel)

    load_coarse_map = load_coarse_map_from_file

    # Make an orthographic projection of the coarse map with (lat, lon) as the centre, and extract from it
    # a raster of dimensions size x size such that one pixel represents extent / size km
    def get_cond_map(self,
        lat: float, lon: float,   # centre of conditioning map as geolocation in degrees
        size: int | None = None,  # size of conditioning map in pixels (always a square)
        extent: float | int | None = None  # distance in km from centre to nearest edge, i.e. radius of inscribed circle
        ) -> np.ndarray:
        if size is None:
            size = self.size
        if extent is None:
            extent = self.extent
        return self._compute_conditioning_map(lat=lat, lon=lon, size=size, extent=extent)

    def get_conditioning_map(self,
        lat: float, lon: float,
        size: int | None = None,
        extent: float | int | None = None
        ) -> np.ndarray:
        if size is None:
            size = self.size
        if extent is None:
            extent = self.extent
        return self._compute_conditioning_map(lat=lat, lon=lon, size=size, extent=extent)

    def _compute_conditioning_map(self, lat: float, lon: float, size: int, extent: float | int) -> np.ndarray:
        # take the coarse_maps numpy array stack and make an orthographic projection of it with centre at (lat, lon)
        # take am square extract of the orthographic projection with (lat, lon) at the centre and extending extent km in each direction
        # rescale to a square raster of dimensions size x size
        # return as a numpy array stack of the same structure as the coarse map
        if not self.coarse_map:
            raise ValueError("No coarse map data loaded. Please load coarse map data first.")

        R = 6371.0  # Earth's spherical radius in km
        dx = 2.0 * extent / size
        x = -extent + (np.arange(size) + 0.5) * dx
        y = extent - (np.arange(size) + 0.5) * dx
        xx, yy = np.meshgrid(x, y)

        phi0 = np.radians(lat)
        lam0 = np.radians(lon)

        rho = np.sqrt(xx**2 + yy**2)
        valid = rho <= R
        rho_safe = np.where(valid, rho, R)
        c = np.arcsin(np.clip(rho_safe / R, 0.0, 1.0))
        cos_c = np.cos(c)

        sin_phi = cos_c * np.sin(phi0) + (yy * np.cos(phi0) / R)
        phi = np.arcsin(np.clip(sin_phi, -1.0, 1.0))

        cos_phi0 = np.cos(phi0)
        sin_phi0 = np.sin(phi0)
        term = R * cos_phi0 * cos_c - yy * sin_phi0
        d_lam = np.arctan2(xx, term)
        lam = lam0 + d_lam

        lats_deg = np.degrees(phi)
        lons_deg = (np.degrees(lam) + 180.0) % 360.0 - 180.0

        projected = {}
        for ch_idx, arr in self.coarse_map.items():
            sampled = sample_equirectangular(arr, lats_deg, lons_deg)
            default_val = -1000.0 if ch_idx == 0 else 0.0
            for name, idx, scale, d_val in CHANNEL_INFO:
                if idx == ch_idx and d_val is not None:
                    default_val = d_val
                    break
            sampled = np.where(valid, sampled, default_val)
            projected[ch_idx] = sampled

        if len(projected) == 1 and 0 in projected:
            return projected[0]
        elif len(projected) == 1:
            return next(iter(projected.values()))
        else:
            sorted_keys = sorted(projected.keys())
            return np.stack([projected[k] for k in sorted_keys], axis=0)