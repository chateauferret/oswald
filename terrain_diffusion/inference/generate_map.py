"""
Generate terrain from conditioning coarse map data in numpy arrays and optionally export to GeoTIFF.

Accepts coarse map data as:
- A 2D numpy array (heightmap / elevation)
- A 3D numpy array shaped (C, H, W) or (H, W, C) where channels correspond to 0..4
- A dictionary mapping channel names ('heightmap', 'temperature', 'temperature_std', 'precipitation', 'precipitation_cv') or indices (0..4) to 2D numpy arrays
- Keyword arguments for individual channels (e.g. heightmap=arr, temperature=arr, etc.)
- A list or tuple of 2D numpy arrays
- A path to a .npy or .npz file (or directory of .npy files)

If `output` is provided (file path), the result is written as a tiled, overview-compressed GeoTIFF.
If `output` is None, the high-resolution generated elevation is returned as a 2D numpy array.

Usage:
  from terrain_diffusion.inference.generate_map import MapGenerator, map_generator, generate_map

  # Class-based usage (loads model once to GPU and generates multiple maps)
  generator = MapGenerator(device="cuda", seed=42)
  elev1 = generator.generate(heightmap1)
  elev2 = generator.generate(heightmap2, seed=123)
  generator.close()  # frees VRAM

  # Or using context manager:
  with MapGenerator(device="cuda", seed=42) as gen:
      elev = gen.generate(heightmap_array)

  # Functional usage:
  elev = generate_map(heightmap_array, seed=42)
"""

import click
import gc
import os
import numpy as np
import rasterio
import torch
from pathlib import Path
from rasterio.transform import Affine
from rasterio.enums import Resampling
from tqdm import tqdm

from terrain_diffusion.paths import get_checkpoint_path, get_data_path
from terrain_diffusion.common.cli_helpers import parse_cache_size
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path

PADDING = 64
PIXELS_PER_CELL = 256

# (channel_name, channel_index, internal_scale, default_value)
# internal_scale: multiplier to convert standard units to pipeline internal units
#   T std (ch 2) is stored as °C×100 internally, so scale=100
# default_value: fill for out-of-bounds conditioning (elevation uses -1000 = deep ocean)
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


def _pad_array(arr: np.ndarray, channel: int, internal_scale: float, default_value: float | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    fill = default_value if default_value is not None else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)

    if internal_scale != 1.0:
        arr = arr * internal_scale

    return np.pad(arr, PADDING, mode="edge")


def _extract_channels(coarse_maps=None, **kwargs) -> dict[int, np.ndarray]:
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


class MapGenerator:
    """Terrain diffusion map generator that loads the model into GPU memory once and supports multiple generations."""

    def __init__(
        self,
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
        endian=None,
        transform=None,
        crs=None,
        scale_temperature_std=True,
        return_array=False,
    ):
        self.model_path = model_path
        self.snr = snr
        self.hdf5_file = hdf5_file
        self.cache_size = cache_size
        self.seed = seed
        self.batch_size = batch_size
        self.torch_compile = torch_compile
        self.dtype = dtype
        self.caching_strategy = caching_strategy
        self.chunk_size = chunk_size
        self.endian = endian
        self.transform = transform
        self.crs = crs
        self.scale_temperature_std = scale_temperature_std
        self.return_array = return_array

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                print("Warning: Using CPU (CUDA not available).")
        else:
            self.device = device

        if self.hdf5_file is not None:
            h_path = resolve_hdf5_path(self.hdf5_file)
            if h_path != 'TEMP' and not Path(h_path).is_absolute():
                h_path = get_data_path(h_path)
            self._resolved_hdf5 = h_path
        else:
            self._resolved_hdf5 = None

        m_path = self.model_path
        if Path(m_path).is_dir() or m_path.startswith('checkpoints/'):
            m_path = get_checkpoint_path(m_path)
        self._resolved_model_path = m_path

        if isinstance(self.batch_size, str):
            self.batch_sizes = [int(x.strip()) for x in self.batch_size.split(",")] if "," in self.batch_size else int(self.batch_size.strip())
        elif isinstance(self.batch_size, (list, tuple)):
            self.batch_sizes = [int(x) for x in self.batch_size]
        elif self.batch_size is not None:
            self.batch_sizes = int(self.batch_size)
        else:
            self.batch_sizes = [1, 4]

        pipeline_dtype = None if self.dtype == "fp32" else self.dtype

        self.world = WorldPipeline.from_pretrained(
            self._resolved_model_path,
            seed=self.seed,
            latents_batch_size=self.batch_sizes,
            torch_compile=self.torch_compile,
            dtype=pipeline_dtype,
            caching_strategy=self.caching_strategy,
            cache_limit=parse_cache_size(self.cache_size),
        )

        self.world.to(self.device)

        if self.snr:
            self._apply_snr(self.snr)

        if self.caching_strategy == "direct":
            self.world.bind(hdf5_file=resolve_hdf5_path(self._resolved_hdf5) if self._resolved_hdf5 else None)
        else:
            self.world.bind(resolve_hdf5_path(self._resolved_hdf5) if self._resolved_hdf5 else "TEMP")

        print(f"MapGenerator initialized on {self.device}. World seed: {self.world.seed}")

    def _apply_snr(self, snr):
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

    def generate(
        self,
        coarse_maps=None,
        output=None,
        return_array=None,
        seed=None,
        snr=None,
        chunk_size=None,
        endian=None,
        transform=None,
        crs=None,
        scale_temperature_std=None,
        **kwargs,
    ):
        """Generate high-resolution terrain for the given coarse map conditioning."""
        if self.world is None:
            raise RuntimeError("MapGenerator model has been closed / deallocated.")

        if return_array is None:
            return_array = self.return_array
        if chunk_size is None:
            chunk_size = self.chunk_size
        if endian is None:
            endian = self.endian
        if transform is None:
            transform = self.transform
        if crs is None:
            crs = self.crs
        if scale_temperature_std is None:
            scale_temperature_std = self.scale_temperature_std

        if seed is not None:
            self.world.change_seed(seed)
        elif self.seed is not None and self.world.seed != self.seed:
            self.world.change_seed(self.seed)

        if snr is not None:
            self._apply_snr(snr)

        # Clear any previously imported custom conditioning channels
        self.world.custom_conditioning_imports.clear()
        self.world.custom_conditioning_import_origins.clear()
        self.world.custom_conditioning_default_values.clear()

        channel_arrays = _extract_channels(coarse_maps, **kwargs)

        H_orig, W_orig = next(iter(channel_arrays.values())).shape
        out_h = H_orig * PIXELS_PER_CELL
        out_w = W_orig * PIXELS_PER_CELL

        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)

        for name, channel, default_scale, default_value in CHANNEL_INFO:
            if channel in channel_arrays:
                arr = channel_arrays[channel]
                internal_scale = default_scale if (channel != 2 or scale_temperature_std) else 1.0
                padded = _pad_array(arr, channel, internal_scale, default_value)
                self.world.set_custom_conditioning_import(channel, padded, 0, 0, default_value=default_value)
                print(f"  Imported {name} → channel {channel}, padded shape: {padded.shape}")
            else:
                print(f"  Skipping {name} (not provided). Perlin noise will be used instead.")

        self.world.rebuild()

        if transform is not None:
            ref_transform = transform
        else:
            ref_transform = Affine.identity()

        out_transform = Affine(
            ref_transform.a / PIXELS_PER_CELL, ref_transform.b, ref_transform.c,
            ref_transform.d, ref_transform.e / PIXELS_PER_CELL, ref_transform.f,
        )
        ref_crs = crs

        if output is not None:
            print(f"Output: {output} ({out_w}x{out_h} px)")

        if chunk_size % PIXELS_PER_CELL != 0:
            raise click.UsageError(f"--chunk-size must be a multiple of {PIXELS_PER_CELL}.")
        chunk_cells = chunk_size // PIXELS_PER_CELL
        row_chunks = (H_orig + chunk_cells - 1) // chunk_cells
        col_chunks = (W_orig + chunk_cells - 1) // chunk_cells

        out_array = None
        if output is None or return_array:
            out_array = np.empty((out_h, out_w), dtype=np.int16)

        dst = None
        if output is not None:
            options = {
                "driver": "GTiff", "height": out_h, "width": out_w,
                "count": 1, "dtype": "int16",
                "crs": ref_crs, "transform": out_transform,
                "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256,
            }
            if endian:
                options["endian"] = endian
            dst = rasterio.open(output, "w", **options)

        try:
            with tqdm(total=row_chunks * col_chunks, desc="Generating") as pbar:
                for ci in range(0, H_orig, chunk_cells):
                    for cj in range(0, W_orig, chunk_cells):
                        ci2 = min(ci + chunk_cells, H_orig)
                        cj2 = min(cj + chunk_cells, W_orig)

                        pi1 = (PADDING + ci) * PIXELS_PER_CELL
                        pi2 = (PADDING + ci2) * PIXELS_PER_CELL
                        pj1 = (PADDING + cj) * PIXELS_PER_CELL
                        pj2 = (PADDING + cj2) * PIXELS_PER_CELL

                        result = self.world.get(pi1, pj1, pi2, pj2, with_climate=False)
                        elev = np.clip(result["elev"].numpy(), -32768, 32767).astype(np.int16)

                        if dst is not None:
                            window = rasterio.windows.Window(
                                cj * PIXELS_PER_CELL, ci * PIXELS_PER_CELL,
                                elev.shape[1], elev.shape[0],
                            )
                            dst.write(elev, 1, window=window)

                        if out_array is not None:
                            out_array[
                                ci * PIXELS_PER_CELL:ci2 * PIXELS_PER_CELL,
                                cj * PIXELS_PER_CELL:cj2 * PIXELS_PER_CELL,
                            ] = elev

                        del result
                        pbar.update(1)
        finally:
            if dst is not None:
                dst.close()

        if output is not None:
            print("Building overviews...")
            with rasterio.open(output, "r+") as dst_ov:
                dst_ov.build_overviews([2, 4, 8, 16, 32, 64], Resampling.bilinear)
                dst_ov.update_tags(ns='rio_overviews', resampling='bilinear')

        # Clean intermediate cached tensors so next generate call starts fresh
        self.world.empty_cache()

        if output is None or return_array:
            return out_array
        return output

    def close(self):
        """Release all model weights, tile cache, and pipeline resources from GPU VRAM and host RAM."""
        if hasattr(self, "world") and self.world is not None:
            try:
                self.world.close()
            except Exception:
                pass
            for attr in ("coarse_model", "base_model", "decoder_model"):
                if hasattr(self.world, attr):
                    setattr(self.world, attr, None)
            self.world = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Alias class name
map_generator = MapGenerator


def generate_map(
    coarse_maps=None,
    output=None,
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
    endian=None,
    transform=None,
    crs=None,
    scale_temperature_std=True,
    return_array=False,
    **kwargs,
):
    """Generate terrain from conditioning numpy arrays using MapGenerator."""
    with MapGenerator(
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
        chunk_size=chunk_size,
        endian=endian,
        transform=transform,
        crs=crs,
        scale_temperature_std=scale_temperature_std,
        return_array=return_array,
    ) as generator:
        return generator.generate(
            coarse_maps=coarse_maps,
            output=output,
            return_array=return_array,
            **kwargs,
        )


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(), required=False, default=None)
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
@click.option("--endian", type=click.Choice(["LITTLE", "BIG"]), default=None, help="Force output endianness (default: system native)")
def main(input_path, output, model_path, snr, hdf5_file, cache_size, seed, device,
         batch_size, torch_compile, dtype, caching_strategy, chunk_size, endian):
    with MapGenerator(
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
        chunk_size=chunk_size,
        endian=endian,
    ) as generator:
        generator.generate(
            coarse_maps=input_path,
            output=output,
        )


if __name__ == "__main__":
    main()
