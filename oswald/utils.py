# (channel_name, channel_index, internal_scale, default_value)
# internal_scale: multiplier to convert standard units to pipeline internal units
#   T std (ch 2) is stored as °C×100 internally, so scale=100
# default_value: fill for out-of-bounds conditioning (elevation uses -1000 = deep ocean)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHANNEL_INFO = [
    ("heightmap",        0, 1.0,   -1000.0),
    ("temperature",      1, 1.0,   None),
    ("temperature_std",  2, 100.0, None),
    ("precipitation",    3, 1.0,   None),
    ("precipitation_cv", 4, 1.0,   None),
]

CHANNEL_ALIASES = {
    "heightmap": 0,
    "elevation": 0,
    "elev": 0,
    "height": 0,
    "dem": 0,
    "temperature": 1,
    "temp": 1,
    "temperature_std": 2,
    "temp_std": 2,
    "temperature_variability": 2,
    "precipitation": 3,
    "precip": 3,
    "rain": 3,
    "precipitation_cv": 4,
    "precip_cv": 4,
    "precipitation_variability": 4,
}

def histogram(
    arr: np.ndarray,
    bins: int | str = 256,
    range: tuple[float, float] | None = None,
    title: str | None = "Histogram",
    xlabel: str | None = "Value",
    ylabel: str | None = "Frequency",
    show: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot and display a histogram of array values in usable form.

    Parameters:
        arr: Input array-like containing numeric data.
        bins: Number of equal-width bins or binning strategy (default 256).
        range: The lower and upper range of the bins. If None, min and max of finite values are used.
        title: Title for the histogram plot. Set to None to omit.
        xlabel: Label for the X-axis. Set to None to omit.
        ylabel: Label for the Y-axis. Set to None to omit.
        show: If True, calls plt.show() to display the plot immediately.
        **kwargs: Additional keyword arguments passed to plt.hist.

    Returns:
        tuple (counts, bin_edges) representing the histogram values and bin edges.
    """
    if hasattr(arr, "detach"):
        data = arr.detach().cpu().numpy()
    else:
        data = np.asarray(arr)

    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        counts = np.array([], dtype=int)
        bin_edges = np.array([0.0, 1.0])
        return counts, bin_edges

    counts, bin_edges = np.histogram(valid_data, bins=bins, range=range)

    plt.figure()
    plt.hist(valid_data.ravel(), bins=bins, range=range, **kwargs)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if show:
        plt.show()

    return counts, bin_edges

def sample_equirectangular(arr: np.ndarray, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
    """Sample values from an equirectangular 2D array using bilinear interpolation.

    The equirectangular map is assumed to cover:
    lat: +90° (row 0) to -90° (row H-1)
    lon: -180° (col 0) to +180° (col W-1)
    """
    H, W = arr.shape
    row = (90.0 - lats_deg) / 180.0 * H - 0.5
    col = (lons_deg + 180.0) / 360.0 * W - 0.5

    r0 = np.floor(row).astype(int)
    r1 = r0 + 1
    dr = row - r0
    r0 = np.clip(r0, 0, H - 1)
    r1 = np.clip(r1, 0, H - 1)

    col_floor = np.floor(col).astype(int)
    dc = col - col_floor
    c0 = col_floor % W
    c1 = (c0 + 1) % W

    return (1.0 - dr) * (1.0 - dc) * arr[r0, c0] + \
           (1.0 - dr) * dc * arr[r0, c1] + \
           dr * (1.0 - dc) * arr[r1, c0] + \
           dr * dc * arr[r1, c1]

def extract_channels(coarse_maps=None, **kwargs) -> dict[int, np.ndarray]:
    """Extract and normalize channel arrays into a dict mapping channel_idx -> 2D numpy array."""
    channels: dict[int, np.ndarray] = {}

    if coarse_maps is not None:
        if isinstance(coarse_maps, (str, Path)):
            path = Path(coarse_maps)
            if path.suffix == ".npy":
                coarse_maps = np.load(path)
            elif path.suffix == ".npz":
                loaded = np.load(path)
                coarse_maps = {k: loaded[k] for k in loaded.files}
            elif path.is_dir():
                dir_dict = {}
                for f in path.glob("*.npy"):
                    dir_dict[f.stem] = np.load(f)
                coarse_maps = dir_dict

        if isinstance(coarse_maps, np.ndarray):
            if coarse_maps.ndim == 2:
                channels[0] = coarse_maps
            elif coarse_maps.ndim == 3:
                # Shape (C, H, W) or (H, W, C)
                if coarse_maps.shape[0] <= 5 and coarse_maps.shape[0] < coarse_maps.shape[1]:
                    for c in range(coarse_maps.shape[0]):
                        channels[c] = coarse_maps[c]
                elif coarse_maps.shape[2] <= 5:
                    for c in range(coarse_maps.shape[2]):
                        channels[c] = coarse_maps[:, :, c]
                else:
                    raise ValueError(f"Unsupported 3D array shape for coarse maps: {coarse_maps.shape}")
            else:
                raise ValueError(f"Coarse map array must be 2D or 3D, got ndim={coarse_maps.ndim}")
        elif isinstance(coarse_maps, dict):
            for k, v in coarse_maps.items():
                if v is None:
                    continue
                if isinstance(k, int):
                    channels[k] = np.asarray(v)
                elif isinstance(k, str):
                    k_lower = k.lower()
                    if k_lower in CHANNEL_ALIASES:
                        channels[CHANNEL_ALIASES[k_lower]] = np.asarray(v)
                    else:
                        try:
                            ch_idx = int(k)
                            channels[ch_idx] = np.asarray(v)
                        except ValueError:
                            raise ValueError(f"Unknown channel name '{k}'. Valid names: {list(CHANNEL_ALIASES.keys())}")
        elif isinstance(coarse_maps, (list, tuple)):
            for c, v in enumerate(coarse_maps):
                if v is not None:
                    channels[c] = np.asarray(v)

    # Also check keyword arguments
    for k, v in kwargs.items():
        if v is None:
            continue
        k_lower = k.lower()
        if k_lower in CHANNEL_ALIASES:
            channels[CHANNEL_ALIASES[k_lower]] = np.asarray(v)

    if not channels:
        raise ValueError("No coarse map data provided. Pass a numpy array, dict of arrays, or channel keyword arguments (e.g. heightmap=arr).")

    # Validate shapes
    shapes = {}
    for ch, arr in channels.items():
        if arr.ndim != 2:
            raise ValueError(f"Channel {ch} array must be 2D, got shape {arr.shape}")
        shapes[ch] = arr.shape

    first_shape = next(iter(shapes.values()))
    for ch, shape in shapes.items():
        if shape != first_shape:
            raise ValueError(f"All coarse map channel arrays must have matching 2D shapes. Got shapes: {shapes}")

    return channels
