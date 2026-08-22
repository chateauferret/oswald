"""
Convert an Azgaar Fantasy Map Builder full JSON export to GeoTIFF rasters
or in-memory conditioning arrays.

The output directory is intended as the conditioning-folder argument to ``tiff-export``
(``python -m terrain_diffusion ... tiff-export <out_dir> output.tif``).

Outputs:
  heightmap.tif        - elevation in meters (float32; uses Azgaar's (h-18)^exponent formula)
  temperature.tif      - mean temperature in °C (float32, from grid cells)
  temperature_std.tif  - temperature std deviation in °C (float32, derived from biome)
  precipitation.tif    - annual precipitation in mm (float32, grid prec * 100)
  precipitation_cv.tif - precipitation coefficient of variation % (float32, derived from biome)

Usage:
  python -m terrain_diffusion.inference.utils.azgaar_to_tiff \
      "Vigny Full.json" output_dir/ --scale 7
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import click
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt, zoom

# Biome ID -> (temp_std_C, precip_cv_pct)
# std and CV are biome-characteristic; actual mean values come from grid cell data.
BIOME_VARIABILITY = {
    0: (float("nan"), float("nan")),  # Marine
    1: (5.0, 80.0),  # Hot Desert
    2: (15.0, 33.0),  # Cold Desert
    3: (5.0, 28.6),  # Savanna
    4: (10.0, 25.0),  # Grassland
    5: (3.0, 26.7),  # Tropical Seasonal Forest
    6: (8.0, 22.2),  # Temperate Deciduous Forest
    7: (2.0, 16.0),  # Tropical Rainforest
    8: (6.0, 25.0),  # Temperate Rainforest
    9: (15.0, 20.0),  # Taiga
    10: (15.0, 25.0),  # Tundra
    11: (10.0, 30.0),  # Glacier
    12: (8.0, 20.0),  # Wetland
}
TEMP_STD_IDX, PRECIP_CV_IDX = 0, 1

# Coarse conditioning cell is 256 native pixels in WorldPipeline / tiff_export.
PIXELS_PER_CELL = 256
TARGET_TILE_WIDTH_KM = 800.0
TARGET_TILE_HEIGHT_KM = 450.0


def load_map(path: str | Path) -> dict[str, Any]:
    """Load an Azgaar Full JSON export into a dict of map components."""
    with open(path) as file:
        data = json.load(file)

    info = data["info"]
    coords = data["mapCoordinates"]
    pack = data["pack"]
    grid = data["grid"]

    pack_verts = {vertex["i"]: vertex["p"] for vertex in pack["vertices"]}
    grid_verts = {vertex["i"]: vertex["p"] for vertex in grid["vertices"]}
    height_exponent = float(data["settings"]["heightExponent"])

    return {
        "map_w": info["width"],
        "map_h": info["height"],
        "coords": coords,
        "pack_cells": pack["cells"],
        "pack_verts": pack_verts,
        "grid_cells": grid["cells"],
        "grid_verts": grid_verts,
        "height_exponent": height_exponent,
        "cells_x": int(grid.get("cellsX") or 0),
        "cells_y": int(grid.get("cellsY") or 0),
    }


def h_to_meters(h, exponent, ocean_max_depth=4000.0, ocean_power=1.5):
    """Convert Azgaar internal height (0-100) to meters.

    Land (h >= 20) matches Azgaar's getHeight(): (h-18)^exponent
    Ocean (h < 20) uses a power curve: -ocean_max_depth * ((20-h)/20)^ocean_power
      h=0  -> -ocean_max_depth (deepest ocean)
      h=19 -> ~-45 m at defaults (coastal shelf)
    """
    if h < 20:
        return -ocean_max_depth * ((20 - h) / 20) ** ocean_power
    return float(h - 18) ** exponent


def build_shapes(cells, verts, scale_x, scale_y, value_fn):
    """Yield (geometry, value) for each cell, using the given vertex lookup."""
    for cell in cells:
        value = value_fn(cell)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            ring = [[px * scale_x, py * scale_y] for px, py in (verts[vi] for vi in cell["v"])]
        except KeyError:
            continue
        yield {"type": "Polygon", "coordinates": [ring]}, value


def rasterize_layer(cells, verts, scale_x, scale_y, shape, value_fn, dtype, fill):
    """Rasterize Voronoi cell polygons into a dense array."""
    shapes = list(build_shapes(cells, verts, scale_x, scale_y, value_fn))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Invalid or empty shape")
        array = rasterize(
            shapes,
            out_shape=shape,
            fill=fill,
            dtype=dtype,
            all_touched=False,
        )
    return array


def fill_nodata(arr, nodata):
    """Replace nodata pixels with the value of the nearest valid pixel."""
    if isinstance(nodata, float) and np.isnan(nodata):
        mask = np.isnan(arr)
    else:
        mask = arr == nodata
    if not mask.any():
        return arr
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(indices)]


def write_tiff(path, arr, transform, crs="EPSG:4326", nodata=None):
    """Write a single-band GeoTIFF."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as destination:
        destination.write(arr, 1)


def rasterize_conditioning_layers(
    map_data: dict[str, Any],
    *,
    out_width: int,
    out_height: int,
    ocean_max_depth: float = 4000.0,
    ocean_power: float = 1.5,
) -> dict[str, np.ndarray]:
    """Rasterize Azgaar Full JSON layers into TIFF-unit float32 arrays.

    Returns keys: elevation_m, temperature_c, temperature_std_c, precipitation_mm,
    precipitation_cv_pct. Arrays are dense with nodata filled from nearest neighbors.
    """
    map_w = map_data["map_w"]
    map_h = map_data["map_h"]
    pack_cells = map_data["pack_cells"]
    pack_verts = map_data["pack_verts"]
    grid_cells = map_data["grid_cells"]
    grid_verts = map_data["grid_verts"]
    height_exponent = map_data["height_exponent"]

    scale_x = out_width / map_w
    scale_y = out_height / map_h
    shape = (out_height, out_width)

    grid_kw = dict(
        cells=grid_cells,
        verts=grid_verts,
        scale_x=scale_x,
        scale_y=scale_y,
        shape=shape,
    )
    pack_kw = dict(
        cells=pack_cells,
        verts=pack_verts,
        scale_x=scale_x,
        scale_y=scale_y,
        shape=shape,
    )

    elevation = rasterize_layer(
        **grid_kw,
        dtype="float32",
        fill=np.nan,
        value_fn=lambda cell: h_to_meters(
            cell.get("h", 0),
            height_exponent,
            ocean_max_depth,
            ocean_power,
        ),
    )
    elevation = fill_nodata(elevation, np.nan).astype(np.float32)

    temperature = rasterize_layer(
        **grid_kw,
        dtype="float32",
        fill=-9999.0,
        value_fn=lambda cell: float(cell["temp"]) if "temp" in cell else None,
    )
    temperature = fill_nodata(temperature, -9999.0).astype(np.float32)

    temperature_std = rasterize_layer(
        **pack_kw,
        dtype="float32",
        fill=-9999.0,
        value_fn=lambda cell: BIOME_VARIABILITY.get(
            cell.get("biome", 0),
            (float("nan"), float("nan")),
        )[TEMP_STD_IDX],
    )
    temperature_std = fill_nodata(temperature_std, -9999.0).astype(np.float32)

    precipitation = rasterize_layer(
        **grid_kw,
        dtype="float32",
        fill=-9999.0,
        value_fn=lambda cell: float(cell["prec"]) * 100.0 if "prec" in cell else None,
    )
    precipitation = fill_nodata(precipitation, -9999.0).astype(np.float32)

    precipitation_cv = rasterize_layer(
        **pack_kw,
        dtype="float32",
        fill=-9999.0,
        value_fn=lambda cell: BIOME_VARIABILITY.get(
            cell.get("biome", 0),
            (float("nan"), float("nan")),
        )[PRECIP_CV_IDX],
    )
    precipitation_cv = fill_nodata(precipitation_cv, -9999.0).astype(np.float32)

    return {
        "elevation_m": elevation,
        "temperature_c": temperature,
        "temperature_std_c": temperature_std,
        "precipitation_mm": precipitation,
        "precipitation_cv_pct": precipitation_cv,
    }


def conditioning_period_cells(
    native_resolution: float,
    *,
    target_width_km: float = TARGET_TILE_WIDTH_KM,
    target_height_km: float = TARGET_TILE_HEIGHT_KM,
) -> tuple[int, int]:
    """Return (period_h, period_w) conditioning cells for an ~800×450 km tile."""
    cell_km = (PIXELS_PER_CELL * float(native_resolution)) / 1000.0
    period_w = max(1, int(round(target_width_km / cell_km)))
    period_h = max(1, int(round(target_height_km / cell_km)))
    return period_h, period_w


def _area_average_resize(array: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Resize with box-style averaging when downsampling, bilinear when upsampling."""
    source = np.asarray(array, dtype=np.float64)
    source_height, source_width = source.shape
    if source_height == out_height and source_width == out_width:
        return source.astype(np.float32)

    zoom_y = out_height / source_height
    zoom_x = out_width / source_width
    # order=1 is adequate; for strong downsampling, pre-bin to nearest integer factors.
    if zoom_y < 1.0 or zoom_x < 1.0:
        factor_y = max(1, int(np.floor(source_height / out_height)))
        factor_x = max(1, int(np.floor(source_width / out_width)))
        if factor_y > 1 or factor_x > 1:
            trimmed_h = (source_height // factor_y) * factor_y
            trimmed_w = (source_width // factor_x) * factor_x
            trimmed = source[:trimmed_h, :trimmed_w]
            binned = trimmed.reshape(
                trimmed_h // factor_y,
                factor_y,
                trimmed_w // factor_x,
                factor_x,
            ).mean(axis=(1, 3))
            source = binned
            source_height, source_width = source.shape
            zoom_y = out_height / source_height
            zoom_x = out_width / source_width

    resized = zoom(source, (zoom_y, zoom_x), order=1)
    # zoom can be off by one pixel due to rounding; crop/pad to exact size.
    resized = np.asarray(resized, dtype=np.float32)
    result = np.zeros((out_height, out_width), dtype=np.float32)
    copy_h = min(out_height, resized.shape[0])
    copy_w = min(out_width, resized.shape[1])
    result[:copy_h, :copy_w] = resized[:copy_h, :copy_w]
    if copy_w < out_width:
        result[:, copy_w:] = result[:, copy_w - 1 : copy_w]
    if copy_h < out_height:
        result[copy_h:, :] = result[copy_h - 1 : copy_h, :]
    return result


def make_periodic_seam(array: np.ndarray, blend: int = 2) -> np.ndarray:
    """Reconcile opposite edges so modular tiling does not show a hard seam.

    Opposite borders are forced equal (required for seamless wrap). Interior
    pixels near the border are eased toward that shared average.
    """
    result = np.asarray(array, dtype=np.float32).copy()
    height, width = result.shape
    blend_x = max(1, min(blend, max(1, width // 4)))
    blend_y = max(1, min(blend, max(1, height // 4)))

    for offset in range(blend_x):
        # alpha=1 at the outer edge, fading to 0 toward the interior.
        alpha = 1.0 - (offset / blend_x)
        left = result[:, offset].copy()
        right = result[:, width - 1 - offset].copy()
        average = 0.5 * (left + right)
        result[:, offset] = (1.0 - alpha) * left + alpha * average
        result[:, width - 1 - offset] = (1.0 - alpha) * right + alpha * average

    for offset in range(blend_y):
        alpha = 1.0 - (offset / blend_y)
        top = result[offset, :].copy()
        bottom = result[height - 1 - offset, :].copy()
        average = 0.5 * (top + bottom)
        result[offset, :] = (1.0 - alpha) * top + alpha * average
        result[height - 1 - offset, :] = (1.0 - alpha) * bottom + alpha * average

    # Exact equality on the wrapped borders (and corners).
    result[:, 0] = result[:, -1] = 0.5 * (result[:, 0] + result[:, -1])
    result[0, :] = result[-1, :] = 0.5 * (result[0, :] + result[-1, :])
    corner = 0.25 * (
        result[0, 0] + result[0, -1] + result[-1, 0] + result[-1, -1]
    )
    result[0, 0] = result[0, -1] = result[-1, 0] = result[-1, -1] = corner
    return result


def to_periodic_conditioning(
    layers: dict[str, np.ndarray],
    period_height: int,
    period_width: int,
    *,
    seam_blend: int = 2,
) -> dict[int, np.ndarray]:
    """Downsample TIFF-unit layers to a seamless periodic conditioning tile.

    Returns WorldPipeline channel index -> float32 array in *internal* units
    (temperature std is scaled to °C×100).
    """
    channel_specs = [
        (0, "elevation_m", 1.0),
        (1, "temperature_c", 1.0),
        (2, "temperature_std_c", 100.0),
        (3, "precipitation_mm", 1.0),
        (4, "precipitation_cv_pct", 1.0),
    ]
    result: dict[int, np.ndarray] = {}
    for channel, key, internal_scale in channel_specs:
        resized = _area_average_resize(layers[key], period_height, period_width)
        seamless = make_periodic_seam(resized, blend=seam_blend)
        result[channel] = (seamless * internal_scale).astype(np.float32)
    return result


def azgaar_json_to_periodic_channels(
    json_path: str | Path,
    native_resolution: float,
    *,
    ocean_max_depth: float = 4000.0,
    ocean_power: float = 1.5,
) -> dict[int, np.ndarray]:
    """Load Azgaar Full JSON and return periodic WorldPipeline conditioning channels."""
    map_data = load_map(json_path)
    # Rasterize near native Voronoi density (cellsX×cellsY), falling back to map px.
    out_width = max(map_data["cells_x"], 64) or int(map_data["map_w"])
    out_height = max(map_data["cells_y"], 64) or int(map_data["map_h"])
    if out_width < 8:
        out_width = int(map_data["map_w"])
    if out_height < 8:
        out_height = int(map_data["map_h"])

    layers = rasterize_conditioning_layers(
        map_data,
        out_width=out_width,
        out_height=out_height,
        ocean_max_depth=ocean_max_depth,
        ocean_power=ocean_power,
    )
    period_height, period_width = conditioning_period_cells(native_resolution)
    return to_periodic_conditioning(layers, period_height, period_width)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--scale", default=100.0, show_default=True, help="Size of each output pixel in km")
@click.option(
    "--ocean-max-depth",
    default=4000.0,
    show_default=True,
    help="Maximum ocean depth in meters (at h=0)",
)
@click.option(
    "--ocean-power",
    default=1.5,
    show_default=True,
    help="Power curve exponent for ocean depth (higher = steeper near coast)",
)
def main(input, output_dir, scale, ocean_max_depth, ocean_power):
    """Convert an Azgaar full JSON export to GeoTIFF rasters."""
    input_path = Path(input)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    map_data = load_map(input_path)
    map_w = map_data["map_w"]
    map_h = map_data["map_h"]
    coords = map_data["coords"]
    print(
        f"  Map size: {map_w}x{map_h}, {len(map_data['grid_cells'])} grid cells, "
        f"{len(map_data['pack_cells'])} pack cells, exponent={map_data['height_exponent']}"
    )

    lon_w, lon_e = coords["lonW"], coords["lonE"]
    lat_s, lat_n = coords["latS"], coords["latN"]

    mid_lat = np.radians((lat_n + lat_s) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mid_lat)
    out_w = max(1, round((lon_e - lon_w) * km_per_deg_lon / scale))
    out_h = max(1, round((lat_n - lat_s) * km_per_deg_lat / scale))
    print(f"  Output shape: {out_w}x{out_h} (WxH), {scale} km/px")

    transform = from_bounds(lon_w, lat_s, lon_e, lat_n, out_w, out_h)
    layers = rasterize_conditioning_layers(
        map_data,
        out_width=out_w,
        out_height=out_h,
        ocean_max_depth=ocean_max_depth,
        ocean_power=ocean_power,
    )

    print("Writing heightmap...")
    write_tiff(output_dir / "heightmap.tif", layers["elevation_m"], transform)
    print(f"  height range: {layers['elevation_m'].min():.0f} .. {layers['elevation_m'].max():.0f} m")

    print("Writing temperature...")
    write_tiff(output_dir / "temperature.tif", layers["temperature_c"], transform)
    print(
        f"  temperature range: {layers['temperature_c'].min():.1f} .. "
        f"{layers['temperature_c'].max():.1f} °C"
    )

    print("Writing temperature std...")
    write_tiff(output_dir / "temperature_std.tif", layers["temperature_std_c"], transform)
    print(
        f"  temperature std range: {layers['temperature_std_c'].min():.1f} .. "
        f"{layers['temperature_std_c'].max():.1f} °C"
    )

    print("Writing precipitation...")
    write_tiff(output_dir / "precipitation.tif", layers["precipitation_mm"], transform)
    print(
        f"  precipitation range: {layers['precipitation_mm'].min():.0f} .. "
        f"{layers['precipitation_mm'].max():.0f} mm/yr"
    )

    print("Writing precipitation CV...")
    write_tiff(output_dir / "precipitation_cv.tif", layers["precipitation_cv_pct"], transform)
    print(
        f"  precipitation CV range: {layers['precipitation_cv_pct'].min():.1f} .. "
        f"{layers['precipitation_cv_pct'].max():.1f} %"
    )

    print(f"\nWrote TIFFs to {output_dir}/")


if __name__ == "__main__":
    main()
