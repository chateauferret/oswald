"""
Export terrain to GeoTIFF from a conditioning TIFF directory.

Typical input is the folder written by ``azgaar-to-tiff`` (``python -m terrain_diffusion
azgaar-to-tiff ...``), using the same channel filenames as ``CHANNEL_FILES`` below. Imports all
conditioning channels with 64-cell edge padding so the model has smooth context at the
borders; padding is stripped from the output.

Usage:
  python -m terrain_diffusion.inference.tiff_export azgaar-output/ output.tif
  python -m terrain_diffusion.inference.tiff_export azgaar-output/ output.tif --snr 1.0,0.5,2.0,0.5,2.0
"""

import click
import os
import numpy as np
import rasterio
import torch
from pathlib import Path
from terrain_diffusion.paths import get_checkpoint_path, get_data_path
from rasterio.transform import Affine
from rasterio.enums import Resampling
from tqdm import tqdm

from terrain_diffusion.common.cli_helpers import parse_cache_size
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path

PADDING = 64
PIXELS_PER_CELL = 256

# (filename, channel_index, internal_scale, default_value)
# internal_scale: multiplier to convert TIFF units to pipeline internal units
#   T std (ch 2) is stored as °C×100 internally but TIFFs are in °C, so scale=100
# default_value: fill for out-of-bounds conditioning (elevation uses -1000 = deep ocean)
CHANNEL_FILES = [
    ("heightmap.tif",        0, 1.0,   -1000.0),
    ("temperature.tif",      1, 1.0,   None),
    ("temperature_std.tif",  2, 100.0, None),
    ("precipitation.tif",    3, 1.0,   None),
    ("precipitation_cv.tif", 4, 1.0,   None),
]



def _load_and_pad(path: Path, channel: int, internal_scale: float, default_value: float | None, byte_swap: bool = False) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        if byte_swap:
            arr = arr.byteswap()
        arr = arr.astype(np.float32)
        nodata = ds.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    fill = default_value if default_value is not None else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)

    if internal_scale != 1.0:
        arr = arr * internal_scale

    return np.pad(arr, PADDING, mode="edge")

def export_tiff(
    tiff_dir,
    output,
    model_path="xandergos/terrain-diffusion-90m",
    snr="0.2,0.2,1.0,0.2,1.0",
    hdf5_file=None,
    cache_size="1G",
    seed=None,
    device=None,
    batch_size="1,4",
    torch_compile=True,
    dtype="fp32",
    caching_strategy="direct",
    chunk_size=8 * 256,
    byte_swap_input=False,
    endian=None,
):

    """Generate terrain from conditioning TIFFs and export to GeoTIFF."""
    tiff_dir = Path(tiff_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if hdf5_file is not None:
        hdf5_file = resolve_hdf5_path(hdf5_file)
        if hdf5_file != 'TEMP' and not Path(hdf5_file).is_absolute():
            hdf5_file = get_data_path(hdf5_file)

    model_path = get_checkpoint_path(model_path) if Path(model_path).is_dir() or model_path.startswith('checkpoints/') else model_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: Using CPU (CUDA not available).")

    if isinstance(batch_size, str):
        batch_sizes = [int(x.strip()) for x in batch_size.split(",")] if "," in batch_size else int(batch_size.strip())
    elif isinstance(batch_size, (list, tuple)):
        batch_sizes = [int(x) for x in batch_size]
    elif batch_size is not None:
        batch_sizes = int(batch_size)
    else:
        batch_sizes = [1, 4]
    if dtype == "fp32":
        dtype = None

    world = WorldPipeline.from_pretrained(
        model_path,
        seed=seed,
        latents_batch_size=batch_sizes,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy=caching_strategy,
        cache_limit=parse_cache_size(cache_size),
    )

    try:
        world.to(device)

        if snr:
            if isinstance(snr, str):
                try:
                    snr_vals = [float(x.strip()) for x in snr.split(",")]
                except ValueError:
                    raise click.UsageError("--snr values must be numbers (e.g. 1.0,0.5,2.0,0.5,2.0).")
            elif isinstance(snr, (list, tuple)):
                snr_vals = [float(x) for x in snr]
            else:
                snr_vals = [float(snr)]
            if len(snr_vals) != 5:
                raise click.UsageError("--snr must have exactly 5 values (e.g. 1.0,0.5,2.0,0.5,2.0).")
            world.set_cond_snr(snr_vals)

        if caching_strategy == "direct":
            world.bind(hdf5_file=resolve_hdf5_path(hdf5_file) if hdf5_file else None)
        else:
            world.bind(resolve_hdf5_path(hdf5_file) if hdf5_file else "TEMP")

        print(f"World seed: {world.seed}")

        ref_transform = None
        ref_crs = None
        H_orig = W_orig = None

        for filename, channel, internal_scale, default_value in CHANNEL_FILES:
            path = tiff_dir / filename
            if not path.exists():
                print(f"  Skipping {filename} (not found). Perlin noise will be used instead.")
                continue

            with rasterio.open(path) as ds:
                if ref_transform is None:
                    ref_transform = ds.transform
                    ref_crs = ds.crs
                    H_orig, W_orig = ds.height, ds.width

            padded = _load_and_pad(path, channel, internal_scale, default_value, byte_swap=byte_swap_input)
            world.set_custom_conditioning_import(channel, padded, 0, 0, default_value=default_value)
            print(f"  Imported {filename} → channel {channel}, padded shape: {padded.shape}")

        if ref_transform is None:
            raise click.UsageError("No conditioning TIFFs found in the directory.")

        out_h = H_orig * PIXELS_PER_CELL
        out_w = W_orig * PIXELS_PER_CELL
        out_transform = Affine(
            ref_transform.a / PIXELS_PER_CELL, ref_transform.b, ref_transform.c,
            ref_transform.d, ref_transform.e / PIXELS_PER_CELL, ref_transform.f,
        )

        print(f"Output: {output} ({out_w}x{out_h} px)")

        if chunk_size % PIXELS_PER_CELL != 0:
            raise click.UsageError(f"--chunk-size must be a multiple of {PIXELS_PER_CELL}.")
        chunk_cells = chunk_size // PIXELS_PER_CELL
        row_chunks = (H_orig + chunk_cells - 1) // chunk_cells
        col_chunks = (W_orig + chunk_cells - 1) // chunk_cells

        with world:
            options = {
                "driver": "GTiff", "height": out_h, "width": out_w,
                "count": 1, "dtype": "int16",
                "crs": ref_crs, "transform": out_transform,
                "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256,
            }
            if endian:
                options["endian"] = endian
            with rasterio.open(output, "w", **options) as dst:
                with tqdm(total=row_chunks * col_chunks, desc="Generating") as pbar:
                    for ci in range(0, H_orig, chunk_cells):
                        for cj in range(0, W_orig, chunk_cells):
                            ci2 = min(ci + chunk_cells, H_orig)
                            cj2 = min(cj + chunk_cells, W_orig)

                            pi1 = (PADDING + ci) * PIXELS_PER_CELL
                            pi2 = (PADDING + ci2) * PIXELS_PER_CELL
                            pj1 = (PADDING + cj) * PIXELS_PER_CELL
                            pj2 = (PADDING + cj2) * PIXELS_PER_CELL

                            result = world.get(pi1, pj1, pi2, pj2, with_climate=False)
                            elev = np.clip(result["elev"].numpy(), -32768, 32767).astype(np.int16)

                            window = rasterio.windows.Window(
                                cj * PIXELS_PER_CELL, ci * PIXELS_PER_CELL,
                                elev.shape[1], elev.shape[0],
                            )
                            dst.write(elev, 1, window=window)

                            # Explicitly free memory for the chunk
                            del result

                            pbar.update(1)

                world.empty_cache()

            # Build overviews for faster lazy loading in viewers
            # Must be done AFTER the file is closed to avoid synchronization issues
            print("Building overviews...")
            with rasterio.open(output, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16, 32, 64], Resampling.bilinear)
                dst.update_tags(ns='rio_overviews', resampling='bilinear')
    finally:
        world.close()

@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.argument("tiff_dir", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--snr",
    metavar="ELEV,TEMP,T_STD,PRECIP,P_CV",
    help=(
        "Conditioning strength per channel (coarse SNR / refinement). "
        "Exactly 5 comma-separated values, e.g. 0.2,0.2,1.0,0.2,1.0"
    ),
    default="0.2,0.2,1.0,0.2,1.0"
)
@click.option("--hdf5-file", default=None, help="HDF5 cache file ('TEMP' for temporary)")
@click.option("--cache-size", default="1G", help="Cache size for direct caching (e.g. 100M, 1G)")
@click.option("--seed", type=int, default=None)
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--batch-size", default="1,4")
@click.option("--compile/--no-compile", "torch_compile", default=True)
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct")
@click.option("--chunk-size", type=int, default=8 * PIXELS_PER_CELL, help="Max query size in output pixels (must be a multiple of 256). Larger values allow bigger batches.")
@click.option("--byte-swap-input", is_flag=True, help="Byte-swap input TIFFs (useful if they have wrong endianness)")
@click.option("--endian", type=click.Choice(["LITTLE", "BIG"]), default=None, help="Force output endianness (default: system native)")
def main(tiff_dir, output, model_path, snr, hdf5_file, cache_size, seed, device,
         batch_size, torch_compile, dtype, caching_strategy, chunk_size, byte_swap_input, endian):
    export_tiff(tiff_dir, output, model_path, snr, hdf5_file, cache_size, seed, device,
         batch_size, torch_compile, dtype, caching_strategy, chunk_size, byte_swap_input, endian)

if __name__ == "__main__":
    main()
