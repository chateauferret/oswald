"""
Convert an ordinary PNG image to a conditioning GeoTIFF for tiff_export.

Treats image luminosity (brightness) as height in meters.
Outputs a float32 heightmap.tif that can be used directly with tiff_export.

Usage:
  python -m terrain_diffusion.inference.utils.png_to_tiff \
      input.png output_dir/ --min-height -1000 --max-height 5000 --km-per-px 1.0
"""

import os
from pathlib import Path
import click
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

def write_tiff(path, arr, transform, crs="EPSG:4326"):
    options = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **options) as dst:
        dst.write(arr.astype(np.float32), 1)

def png_to_tiff(input_image, output_dir, min_height=-1000.0, max_height=8000.0, km_per_px=1.0, lat=0.0, lon=0.0):
    """Convert a PNG image to a heightmap TIFF based on luminosity.

    Args:
        input_image: Path to the input PNG image.
        output_dir: Directory where heightmap.tif will be written.
        min_height: Height in meters mapped to black (0).
        max_height: Height in meters mapped to white (255).
        km_per_px: Resolution in km/pixel.
        lat: Center latitude for the output geotransform.
        lon: Center longitude for the output geotransform.
    """
    input_path = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    img = Image.open(input_path).convert("L")
    arr = np.array(img).astype(np.float32)

    h, w = arr.shape
    print(f"  Image size: {w}x{h}")

    # Map 0-255 to min_height - max_height
    heights = (arr / 255.0) * (max_height - min_height) + min_height

    # Calculate bounds based on km_per_px
    # 1 degree lat approx 111.32 km
    # 1 degree lon approx 111.32 * cos(lat) km
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat))

    deg_w = (w * km_per_px) / km_per_deg_lon
    deg_h = (h * km_per_px) / km_per_deg_lat

    lon_w = lon - deg_w / 2
    lon_e = lon + deg_w / 2
    lat_s = lat - deg_h / 2
    lat_n = lat + deg_h / 2

    transform = from_bounds(lon_w, lat_s, lon_e, lat_n, w, h)

    out_file = output_dir / "heightmap.tif"
    print(f"Writing {out_file}...")
    write_tiff(out_file, heights, transform)

    print(f"Done. Height range: {heights.min():.1f} to {heights.max():.1f} meters")


@click.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--min-height", default=-1000.0, show_default=True, help="Height in meters mapped to black (0)")
@click.option("--max-height", default=8000.0, show_default=True, help="Height in meters mapped to white (255)")
@click.option("--km-per-px", default=1.0, show_default=True, help="Resolution in km/pixel")
@click.option("--lat", default=0.0, show_default=True, help="Center latitude for the output geotransform")
@click.option("--lon", default=0.0, show_default=True, help="Center longitude for the output geotransform")
def main(input_image, output_dir, min_height, max_height, km_per_px, lat, lon):
    """Convert a PNG image to a heightmap TIFF based on luminosity."""
    png_to_tiff(input_image, output_dir, min_height, max_height, km_per_px, lat, lon)

if __name__ == "__main__":
    main()
