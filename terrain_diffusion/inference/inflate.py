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


class TiffFactory:
    """
    A class to perform terrain inference and export TIFFs from a set of conditioning world maps.
    Constructed with a directory containing conditioning TIFFs, it allows exporting subsets
    of the high-resolution generated terrain.
    """
    def __init__(
        self,
        tiff_dir,
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
        byte_swap_input=False,
    ):
        self.tiff_dir = Path(tiff_dir)

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

        self.world = WorldPipeline.from_pretrained(
            model_path,
            seed=seed,
            latents_batch_size=batch_sizes,
            torch_compile=torch_compile,
            dtype=dtype,
            caching_strategy=caching_strategy,
            cache_limit=parse_cache_size(cache_size),
        )
        self.world.to(device)

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
            self.world.set_cond_snr(snr_vals)

        if caching_strategy == "direct":
            self.world.bind(hdf5_file=resolve_hdf5_path(hdf5_file) if hdf5_file else None)
        else:
            self.world.bind(resolve_hdf5_path(hdf5_file) if hdf5_file else "TEMP")

        print(f"World seed: {self.world.seed}")

        self.ref_transform = None
        self.ref_crs = None
        self.H_orig = self.W_orig = None

        for filename, channel, internal_scale, default_value in CHANNEL_FILES:
            path = self.tiff_dir / filename
            if not path.exists():
                print(f"  Skipping {filename} (not found). Perlin noise will be used instead.")
                continue

            with rasterio.open(path) as ds:
                if self.ref_transform is None:
                    self.ref_transform = ds.transform
                    self.ref_crs = ds.crs
                    self.H_orig, self.W_orig = ds.height, ds.width

            padded = _load_and_pad(path, channel, internal_scale, default_value, byte_swap=byte_swap_input)
            self.world.set_custom_conditioning_import(channel, padded, 0, 0, default_value=default_value)
            print(f"  Imported {filename} → channel {channel}, padded shape: {padded.shape}")

        if self.ref_transform is None:
            raise click.UsageError("No conditioning TIFFs found in the directory.")

    def export_subset(
        self,
        output,
        ci=0, cj=0, ci2=None, cj2=None,
        chunk_size=8 * 256,
        endian=None,
    ):
        """Export a subset of the high-resolution terrain to a GeoTIFF."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if ci2 is None: ci2 = self.H_orig
        if cj2 is None: cj2 = self.W_orig

        H_sub = ci2 - ci
        W_sub = cj2 - cj

        out_h = H_sub * PIXELS_PER_CELL
        out_w = W_sub * PIXELS_PER_CELL
        
        subset_origin_x, subset_origin_y = self.ref_transform * (cj, ci)
        
        out_transform = Affine(
            self.ref_transform.a / PIXELS_PER_CELL, self.ref_transform.b, subset_origin_x,
            self.ref_transform.d, self.ref_transform.e / PIXELS_PER_CELL, subset_origin_y,
        )

        print(f"Output: {output} ({out_w}x{out_h} px)")

        if chunk_size % PIXELS_PER_CELL != 0:
            raise click.UsageError(f"--chunk-size must be a multiple of {PIXELS_PER_CELL}.")
        
        chunk_cells = chunk_size // PIXELS_PER_CELL
        row_chunks = (H_sub + chunk_cells - 1) // chunk_cells
        col_chunks = (W_sub + chunk_cells - 1) // chunk_cells

        with self.world:
            options = {
                "driver": "GTiff", "height": out_h, "width": out_w,
                "count": 1, "dtype": "int16",
                "crs": self.ref_crs, "transform": out_transform,
                "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256,
            }
            if endian:
                options["endian"] = endian
            with rasterio.open(output, "w", **options) as dst:
                with tqdm(total=row_chunks * col_chunks, desc="Generating") as pbar:
                    for i in range(ci, ci2, chunk_cells):
                        for j in range(cj, cj2, chunk_cells):
                            curr_i2 = min(i + chunk_cells, ci2)
                            curr_j2 = min(j + chunk_cells, cj2)

                            pi1 = (PADDING + i) * PIXELS_PER_CELL
                            pi2 = (PADDING + curr_i2) * PIXELS_PER_CELL
                            pj1 = (PADDING + j) * PIXELS_PER_CELL
                            pj2 = (PADDING + curr_j2) * PIXELS_PER_CELL

                            result = self.world.get(pi1, pj1, pi2, pj2, with_climate=False)
                            elev = np.clip(result["elev"].numpy(), -32768, 32767).astype(np.int16)

                            window = rasterio.windows.Window(
                                (j - cj) * PIXELS_PER_CELL, (i - ci) * PIXELS_PER_CELL,
                                elev.shape[1], elev.shape[0],
                            )
                            dst.write(elev, 1, window=window)

                            del result
                            pbar.update(1)

            self.world.empty_cache()

            print("Building overviews...")
            with rasterio.open(output, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16, 32, 64], Resampling.bilinear)
                dst.update_tags(ns='rio_overviews', resampling='bilinear')

    def close(self):
        """Release resources associated with the pipeline."""
        self.world.close()



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
    factory = TiffFactory(
        tiff_dir=tiff_dir,
        model_path=model_path,
        snr=snr,
        hdf5_file=hdf5_file,
        cache_size=cache_size,
        seed=seed,
        device=device,
        batch_size=batch_size,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy=caching_strategy,
        byte_swap_input=byte_swap_input,
    )
    try:
        factory.export_subset(
            output=output,
            chunk_size=chunk_size,
            endian=endian,
        )
    finally:
        factory.close()

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
