"""
REST server for terrain diffusion from a given set of coarse maps.

Server provides the following interfaces:
    - POST /coarse-maps/<channel>: Accept and store the world's coarse map for a given channel.
    - POST /jobs/?lat=<lat>&lon=<lon>&size=<size>&extent=<extent>:
        Enqueue a terrain diffusion job using as conditioning map the extract specified by the geometry parameters.
        Return a unique job identifier.
    - GET /status/<job-id>: Get the status of the specified job.
    - GET /fine-maps/<job-id>/<channel>:
        Get the generated fine map for a given channel, as geotiff with its WKT orthographic projection string.

"""

import io
import os
import sys
import time
import uuid
import queue
import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import click
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.io import MemoryFile
from PIL import Image

from flask import Flask, request, jsonify, send_file, Response, make_response

from oswald.coarse_map import CoarseMap
from oswald.generate_map import MapGenerator, generate_map, PIXELS_PER_CELL
from oswald.utils import CHANNEL_INFO, CHANNEL_ALIASES

logger = logging.getLogger("diffusion_server")

EARTH_RADIUS_M = 6371000.0  # Earth's spherical radius in meters (matching coarse_map.py 6371 km)


def create_orthographic_crs(lat: float, lon: float) -> CRS:
    """Create an orthographic CRS centered at (lat, lon) with Earth radius 6371km."""
    proj_str = f"+proj=ortho +lat_0={float(lat)} +lon_0={float(lon)} +x_0=0 +y_0=0 +R={int(EARTH_RADIUS_M)} +units=m +no_defs"
    return CRS.from_proj4(proj_str)


def compute_orthographic_wkt(lat: float, lon: float) -> str:
    """Compute the WKT projection string for an orthographic projection centered at (lat, lon)."""
    crs = create_orthographic_crs(lat, lon)
    return crs.to_wkt()


def _parse_channel_identifier(channel: Union[str, int]) -> int:
    """Resolve a channel name or index to an integer channel ID (0..4)."""
    if isinstance(channel, str):
        ch_lower = channel.lower().strip()
        if ch_lower in CHANNEL_ALIASES:
            return CHANNEL_ALIASES[ch_lower]
        try:
            return int(ch_lower)
        except ValueError:
            raise ValueError(f"Unknown channel identifier: '{channel}'. Valid channels: {list(CHANNEL_ALIASES.keys())}")
    elif isinstance(channel, int):
        return channel
    else:
        raise ValueError(f"Invalid channel type: {type(channel)}")


def _load_array_from_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """Load a numpy array from in-memory bytes of various formats (GeoTIFF, NPY, NPZ, PNG, etc.)."""
    # 1. Try rasterio (GeoTIFF / TIFF)
    try:
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                arr = src.read(1)
                return np.asarray(arr, dtype=np.float32)
    except Exception:
        pass

    # 2. Try numpy NPY / NPZ
    try:
        buf = io.BytesIO(data)
        loaded = np.load(buf)
        if isinstance(loaded, np.ndarray):
            return np.asarray(loaded, dtype=np.float32)
        elif hasattr(loaded, "files") and len(loaded.files) > 0:
            return np.asarray(loaded[loaded.files[0]], dtype=np.float32)
    except Exception:
        pass

    # 3. Try PIL Image
    try:
        buf = io.BytesIO(data)
        img = Image.open(buf)
        return np.asarray(img, dtype=np.float32)
    except Exception:
        pass

    raise ValueError("Could not decode coarse map from uploaded binary data.")


class DiffusionServer:
    """REST server managing coarse map reference data and background terrain diffusion jobs."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        model_path: str = "xandergos/terrain-diffusion-90m",
        device: Optional[str] = None,
        coarse_map: Optional[CoarseMap] = None,
        generator: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
        num_workers: int = 1,
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.device = device
        self.coarse_map = coarse_map if coarse_map is not None else CoarseMap()
        self.generator = generator
        self.lazy_load = lazy_load
        self.num_workers = num_workers

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self._tmp_dir = None
        else:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="diffusion_server_")
            self.output_dir = Path(self._tmp_dir.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_threads: list[threading.Thread] = []

        # Initialize background worker threads
        self._start_workers()

        # Build Flask application
        self.app = create_app(server=self)

    def _start_workers(self):
        """Start background worker threads for job execution."""
        self._stop_event.clear()
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"diffusion-worker-{i}")
            t.start()
            self._worker_threads.append(t)

    def _worker_loop(self):
        """Background worker thread processing terrain diffusion jobs from the queue."""
        while not self._stop_event.is_set():
            try:
                job_id = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                job = self.jobs.get(job_id)

            if job is None:
                self.job_queue.task_done()
                continue

            try:
                self._execute_job(job)
            except Exception as e:
                logger.exception(f"Job {job_id} failed with error: {e}")
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = str(e)
                    job["completed_at"] = time.time()
            finally:
                self.job_queue.task_done()

    def _execute_job(self, job: Dict[str, Any]):
        """Execute a single terrain diffusion job and write output GeoTIFF."""
        job_id = job["job_id"]
        lat = float(job["lat"])
        lon = float(job["lon"])
        size = int(job["size"])
        extent = float(job["extent"])

        with self._lock:
            job["status"] = "running"
            job["started_at"] = time.time()

        # 1. Extract conditioning coarse map
        cond_map = self.coarse_map.get_cond_map(lat=lat, lon=lon, size=size, extent=extent)

        # 2. Compute projection and transforms
        crs = create_orthographic_crs(lat, lon)
        wkt = crs.to_wkt()

        extent_m = extent * 1000.0
        dx_coarse_m = 2.0 * extent_m / size
        ref_transform = from_origin(-extent_m, extent_m, dx_coarse_m, dx_coarse_m)

        out_transform = Affine(
            ref_transform.a / PIXELS_PER_CELL, ref_transform.b, ref_transform.c,
            ref_transform.d, ref_transform.e / PIXELS_PER_CELL, ref_transform.f,
        )

        job_out_dir = self.output_dir / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)
        output_tiff = job_out_dir / "heightmap.tif"

        # 3. Generate high-resolution fine map
        if self.generator is not None:
            if hasattr(self.generator, "generate"):
                self.generator.generate(
                    coarse_maps=cond_map,
                    output=output_tiff,
                    crs=crs,
                    transform=ref_transform,
                    seed=job.get("seed"),
                    snr=job.get("snr"),
                )
            elif callable(self.generator):
                self.generator(
                    coarse_maps=cond_map,
                    output=output_tiff,
                    crs=crs,
                    transform=ref_transform,
                    seed=job.get("seed"),
                    snr=job.get("snr"),
                )
        else:
            generate_map(
                coarse_maps=cond_map,
                output=output_tiff,
                crs=crs,
                transform=ref_transform,
                model_path=self.model_path,
                device=self.device,
                seed=job.get("seed"),
                snr=job.get("snr"),
            )

        # 4. Save results to job state
        with self._lock:
            job["wkt"] = wkt
            job["crs"] = str(crs)
            job["transform"] = out_transform
            job["output_files"] = {
                "heightmap": str(output_tiff),
                "elevation": str(output_tiff),
                "elev": str(output_tiff),
                "0": str(output_tiff),
                0: str(output_tiff),
            }
            job["status"] = "completed"
            job["completed_at"] = time.time()

    def add_coarse_map(self, channel: Union[str, int], data: Any) -> int:
        """Store reference coarse map for a channel. Returns integer channel index."""
        ch_idx = _parse_channel_identifier(channel)
        if isinstance(data, bytes):
            arr = _load_array_from_bytes(data)
            self.coarse_map.add_channel(coarse_map=arr, channel=ch_idx)
        elif isinstance(data, (np.ndarray, Image.Image, str, Path)):
            self.coarse_map.add_channel(coarse_map=data, channel=ch_idx)
        elif isinstance(data, (list, tuple)):
            arr = np.asarray(data, dtype=np.float32)
            self.coarse_map.add_channel(coarse_map=arr, channel=ch_idx)
        else:
            raise ValueError(f"Unsupported coarse map data type: {type(data)}")
        return ch_idx

    def enqueue_job(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        size: int = 64,
        extent: Union[float, int] = 3000.0,
        seed: Optional[int] = None,
        snr: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """Enqueue a terrain diffusion job and return unique job ID."""
        if not self.coarse_map.coarse_map:
            raise ValueError("No coarse map reference data loaded. Please upload coarse maps first.")

        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "id": job_id,
            "status": "pending",
            "lat": float(lat),
            "lon": float(lon),
            "size": int(size),
            "extent": float(extent),
            "seed": seed,
            "snr": snr,
            "wkt": compute_orthographic_wkt(float(lat), float(lon)),
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_files": {},
            "kwargs": kwargs,
        }

        with self._lock:
            self.jobs[job_id] = job

        self.job_queue.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job record dictionary."""
        with self._lock:
            return self.jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get public status dictionary for a job."""
        job = self.get_job(job_id)
        if job is None:
            return None

        status_dict = {
            "job_id": job["job_id"],
            "id": job["job_id"],
            "status": job["status"],
            "lat": job["lat"],
            "lon": job["lon"],
            "size": job["size"],
            "extent": job["extent"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "wkt": job.get("wkt"),
        }
        if job.get("error"):
            status_dict["error"] = job["error"]
        if job.get("output_files"):
            status_dict["channels"] = list(job["output_files"].keys())
        return status_dict

    def get_fine_map(self, job_id: str, channel: Union[str, int]) -> Tuple[Path, str]:
        """Get the generated fine map file path and its WKT projection string."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job '{job_id}' not found.")
        if job["status"] != "completed":
            raise RuntimeError(f"Job '{job_id}' is not completed (status: {job['status']}).")

        ch_idx = _parse_channel_identifier(channel)
        output_files = job.get("output_files", {})

        file_path = None
        if ch_idx in output_files:
            file_path = output_files[ch_idx]
        elif str(channel).lower() in output_files:
            file_path = output_files[str(channel).lower()]
        elif "heightmap" in output_files and ch_idx == 0:
            file_path = output_files["heightmap"]

        if file_path is None or not Path(file_path).exists():
            raise FileNotFoundError(f"Fine map for channel '{channel}' not found for job '{job_id}'.")

        wkt = job.get("wkt", compute_orthographic_wkt(job["lat"], job["lon"]))
        return Path(file_path), wkt

    def process_job_sync(self, job_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for a job to complete synchronously."""
        start = time.time()
        while time.time() - start < timeout:
            job = self.get_job(job_id)
            if job and job["status"] in ("completed", "failed"):
                return job
            time.sleep(0.05)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s.")

    def close(self):
        """Shutdown worker threads and cleanup temporary resources."""
        self._stop_event.set()
        for t in self._worker_threads:
            t.join(timeout=2.0)
        self._worker_threads.clear()

        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except Exception:
                pass
            self._tmp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: bool = False, **kwargs):
        """Run the Flask REST server."""
        h = host or self.host
        p = port or self.port
        self.app.run(host=h, port=p, debug=debug, **kwargs)


# Alias class name
diffusion_server = DiffusionServer


def create_app(
    server: Optional[DiffusionServer] = None,
    coarse_map: Optional[CoarseMap] = None,
    generator: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_path: str = "xandergos/terrain-diffusion-90m",
    device: Optional[str] = None,
) -> Flask:
    """Create and configure Flask application for the terrain diffusion REST API."""
    app = Flask(__name__)

    if server is None:
        server = DiffusionServer(
            coarse_map=coarse_map,
            generator=generator,
            output_dir=output_dir,
            model_path=model_path,
            device=device,
        )

    app.config["DIFFUSION_SERVER"] = server

    # -------------------------------------------------------------------------
    # Route: POST /coarse-maps/<channel>
    # -------------------------------------------------------------------------
    @app.route("/coarse-maps/<channel>", methods=["POST"])
    @app.route("/coarse_maps/<channel>", methods=["POST"])
    def post_coarse_map(channel: str):
        """Accept and store the world's coarse map for a given channel."""
        srv: DiffusionServer = app.config["DIFFUSION_SERVER"]
        try:
            ch_idx = _parse_channel_identifier(channel)
        except ValueError as e:
            return jsonify({"error": str(e), "status": "error"}), 400

        try:
            # 1. Check for file uploads
            if request.files:
                file_obj = next(iter(request.files.values()))
                file_bytes = file_obj.read()
                srv.add_coarse_map(ch_idx, file_bytes)
                return jsonify({
                    "status": "ok",
                    "channel": channel,
                    "channel_index": ch_idx,
                    "message": f"Coarse map for channel '{channel}' stored successfully.",
                }), 200

            # 2. Check for JSON payload
            if request.is_json or request.content_type == "application/json":
                data = request.get_json(silent=True) or {}
                if "path" in data or "file_path" in data:
                    fpath = data.get("path") or data.get("file_path")
                    srv.add_coarse_map(ch_idx, fpath)
                elif "array" in data:
                    arr = np.asarray(data["array"], dtype=np.float32)
                    srv.add_coarse_map(ch_idx, arr)
                elif "data" in data and isinstance(data["data"], (list, tuple)):
                    arr = np.asarray(data["data"], dtype=np.float32)
                    srv.add_coarse_map(ch_idx, arr)
                else:
                    return jsonify({"error": "JSON body must contain 'file_path', 'path', or 'array'", "status": "error"}), 400

                return jsonify({
                    "status": "ok",
                    "channel": channel,
                    "channel_index": ch_idx,
                    "message": f"Coarse map for channel '{channel}' stored successfully.",
                }), 200

            # 3. Check for form data (file_path / path)
            if request.form and ("file_path" in request.form or "path" in request.form):
                fpath = request.form.get("file_path") or request.form.get("path")
                srv.add_coarse_map(ch_idx, fpath)
                return jsonify({
                    "status": "ok",
                    "channel": channel,
                    "channel_index": ch_idx,
                    "message": f"Coarse map for channel '{channel}' stored successfully.",
                }), 200

            # 4. Check for raw body bytes
            raw_data = request.get_data()
            if raw_data:
                srv.add_coarse_map(ch_idx, raw_data)
                return jsonify({
                    "status": "ok",
                    "channel": channel,
                    "channel_index": ch_idx,
                    "message": f"Coarse map for channel '{channel}' stored successfully.",
                }), 200

            return jsonify({"error": "No coarse map data provided in request.", "status": "error"}), 400

        except Exception as e:
            logger.exception("Error in post_coarse_map")
            return jsonify({"error": str(e), "status": "error"}), 400

    # -------------------------------------------------------------------------
    # Route: POST /jobs/
    # -------------------------------------------------------------------------
    @app.route("/jobs", methods=["POST"])
    @app.route("/jobs/", methods=["POST"])
    def post_job():
        """Enqueue a terrain diffusion job using conditioning extract specified by geometry parameters."""
        srv: DiffusionServer = app.config["DIFFUSION_SERVER"]

        # Parse query params, JSON, or form fields
        args = request.args
        json_data = request.get_json(silent=True) or {}
        form_data = request.form

        def _get_val(key, default=None):
            if key in args:
                return args[key]
            if key in json_data:
                return json_data[key]
            if key in form_data:
                return form_data[key]
            return default

        try:
            lat_raw = _get_val("lat", 0.0)
            lon_raw = _get_val("lon", 0.0)
            size_raw = _get_val("size", 64)
            extent_raw = _get_val("extent", 3000.0)

            lat = float(lat_raw)
            lon = float(lon_raw)
            size = int(size_raw)
            extent = float(extent_raw)

            seed_val = _get_val("seed", None)
            seed = int(seed_val) if seed_val is not None else None

            snr = _get_val("snr", None)

            if size <= 0:
                return jsonify({"error": "Parameter 'size' must be a positive integer.", "status": "error"}), 400
            if extent <= 0:
                return jsonify({"error": "Parameter 'extent' must be a positive number.", "status": "error"}), 400

            job_id = srv.enqueue_job(
                lat=lat,
                lon=lon,
                size=size,
                extent=extent,
                seed=seed,
                snr=snr,
            )

            return jsonify({
                "job_id": job_id,
                "id": job_id,
                "status": "pending",
                "message": "Job enqueued successfully.",
                "lat": lat,
                "lon": lon,
                "size": size,
                "extent": extent,
            }), 200

        except ValueError as e:
            return jsonify({"error": str(e), "status": "error"}), 400
        except Exception as e:
            logger.exception("Error in post_job")
            return jsonify({"error": str(e), "status": "error"}), 400

    # -------------------------------------------------------------------------
    # Route: GET /status/<job_id>
    # -------------------------------------------------------------------------
    @app.route("/status/<job_id>", methods=["GET"])
    @app.route("/status/<job_id>/", methods=["GET"])
    @app.route("/jobs/<job_id>", methods=["GET"])
    @app.route("/jobs/<job_id>/status", methods=["GET"])
    def get_status(job_id: str):
        """Get the status of the specified job."""
        srv: DiffusionServer = app.config["DIFFUSION_SERVER"]
        status_info = srv.get_job_status(job_id)
        if status_info is None:
            return jsonify({"error": f"Job '{job_id}' not found.", "status": "not_found"}), 404
        return jsonify(status_info), 200

    # -------------------------------------------------------------------------
    # Route: GET /fine-maps/<job_id>/<channel>
    # -------------------------------------------------------------------------
    @app.route("/fine-maps/<job_id>/<channel>", methods=["GET"])
    @app.route("/fine-maps/<job_id>/<channel>/", methods=["GET"])
    @app.route("/fine_maps/<job_id>/<channel>", methods=["GET"])
    @app.route("/fine_maps/<job_id>/<channel>/", methods=["GET"])
    def get_fine_map_endpoint(job_id: str, channel: str):
        """Get the generated fine map for a given channel as GeoTIFF with WKT orthographic projection."""
        srv: DiffusionServer = app.config["DIFFUSION_SERVER"]
        job = srv.get_job(job_id)
        if job is None:
            return jsonify({"error": f"Job '{job_id}' not found.", "status": "not_found"}), 404

        if job["status"] in ("pending", "running"):
            return jsonify({
                "job_id": job_id,
                "status": job["status"],
                "message": f"Job '{job_id}' is still {job['status']}.",
            }), 202

        if job["status"] == "failed":
            return jsonify({
                "job_id": job_id,
                "status": "failed",
                "error": job.get("error", "Job failed."),
            }), 500

        try:
            file_path, wkt = srv.get_fine_map(job_id, channel)
        except (KeyError, FileNotFoundError, ValueError) as e:
            return jsonify({"error": str(e), "status": "error"}), 404
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

        # Build response with GeoTIFF and WKT header
        response = make_response(send_file(
            file_path,
            mimetype="image/tiff",
            as_attachment=True,
            download_name=f"fine_map_{job_id}_{channel}.tif",
        ))
        response.headers["X-Projection-WKT"] = wkt
        response.headers["X-Job-ID"] = job_id
        response.headers["X-Channel"] = str(channel)
        return response

    return app


# Default module-level WSGI application
app = create_app()


@click.command()
@click.option("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
@click.option("--port", default=5000, type=int, help="Port to bind (default: 5000)")
@click.option("--model-path", default="xandergos/terrain-diffusion-90m", help="Path or HuggingFace ID of model")
@click.option("--device", default=None, help="Device to use ('cuda' or 'cpu')")
@click.option("--debug/--no-debug", default=False, help="Enable Flask debug mode")
def main(host: str, port: int, model_path: str, device: Optional[str], debug: bool):
    """Run the Oswald Terrain Diffusion REST Server."""
    server = DiffusionServer(host=host, port=port, model_path=model_path, device=device)
    print(f"Starting Oswald Terrain Diffusion Server on http://{host}:{port}")
    try:
        server.run(host=host, port=port, debug=debug)
    finally:
        server.close()


if __name__ == "__main__":
    main()
