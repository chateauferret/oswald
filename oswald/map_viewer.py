import io
from pathlib import Path
from typing import Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from IPython.display import display
from matplotlib.colors import Colormap
from PIL import Image
from rasterio.enums import Resampling

from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.inference.utils.tiff_stats import calculate_stats


class MapViewer:
    def __init__(
        self,
        path: Union[str, Path, np.ndarray, None] = None,
        cmap: Union[str, Colormap] = "terrain",
        display_size: int = 1024,
        resolution: float = 90.0,
        byte_swap: bool = False,
        vmin: float = -4000.0,
        vmax: float = 4000.0,
        data: Optional[np.ndarray] = None,
        array: Optional[np.ndarray] = None,
    ):
        self.btn_stats = None
        self.btn_zoom_in = None
        self.btn_zoom_out = None
        self.slider_relief = None
        self.check_relief = None
        self.btn_up = None
        self.btn_down = None
        self.btn_left = None
        self.btn_right = None
        self.cmap = cmap
        self.display_size = display_size
        self.resolution = resolution
        self.byte_swap = byte_swap
        self.vmin = vmin if vmin is not None else -4000.0
        self.vmax = vmax if vmax is not None else 4000.0

        source = path if path is not None else (data if data is not None else array)
        if source is None:
            raise ValueError("Must provide either a file path or a NumPy array to TiffViewer.")

        if isinstance(source, np.ndarray) or isinstance(source, Image.Image):
            arr = np.asarray(source, dtype=np.float32)
            if arr.ndim == 3:
                if arr.shape[0] <= 5:
                    arr = arr[0]
                elif arr.shape[2] <= 5:
                    arr = arr[:, :, 0]
                else:
                    arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array for elevation, got shape {arr.shape}")

            self.data = arr
            self.path = None
            self.ds = None
            self.width = int(self.data.shape[1])
            self.height = int(self.data.shape[0])
        else:
            self.path = str(source)
            self.ds = rasterio.open(self.path)
            self.data = None
            self.width = self.ds.width
            self.height = self.ds.height

        # Current view state: center x, y in pixels, and zoom level
        # zoom=1.0 means the whole image is display_size pixels wide
        # zoom=width/display_size means 1:1 pixel mapping
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.zoom = 1.0

        self.show_relief = False
        self.relief_intensity = 1.5
        self.relief_resolution_scale = 0.25

        # Widgets
        self.img_widget = widgets.Image(format='png', width=display_size, height=display_size)
        self.status_label = widgets.Label()

    def _get_window_and_scale(self) -> rasterio.windows.Window:
        # How many TIFF/array pixels to show?
        view_w = self.width / self.zoom
        view_h = view_w  # square display

        w = rasterio.windows.Window(
            self.cx - view_w / 2.0,
            self.cy - view_h / 2.0,
            view_w,
            view_h,
        )
        return w

    def _update_image(self):
        window = self._get_window_and_scale()

        # Read data
        if self.ds is not None:
            data = self.ds.read(
                1,
                window=window,
                out_shape=(self.display_size, self.display_size),
                resampling=Resampling.bilinear,
                boundless=True,
                out_dtype=np.float32,
            )
        else:
            col_off = window.col_off
            row_off = window.row_off
            width = window.width
            height = window.height

            pil_img = Image.fromarray(self.data)
            extent = (col_off, row_off, col_off + width, row_off + height)
            transformed = pil_img.transform(
                (self.display_size, self.display_size),
                Image.Transform.EXTENT,
                extent,
                resample=Image.Resampling.BILINEAR,
            )
            data = np.asarray(transformed, dtype=np.float32)

        if self.byte_swap:
            data = data.byteswap()

        rgb = self._colorize(data)

        if self.show_relief:
            effective_resolution = (
                self.resolution
                * (window.width / self.display_size)
                * self.relief_resolution_scale
            )
            shaded = get_relief_map(
                data, None, None, None,
                resolution=effective_resolution,
                relief=self.relief_intensity,
                vmin=self.vmin,
                vmax=self.vmax,
                rgb=rgb,
            )
            # shaded is (H, W, 3) float 0..1
            img_data = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
        else:
            img_data = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        # Convert to PNG
        pil_img = Image.fromarray(img_data)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        self.img_widget.value = buf.getvalue()

        self.status_label.value = (
            f"Center: ({int(self.cx)}, {int(self.cy)}) | "
            f"Zoom: {self.zoom:.2f}x | "
            f"Window: {int(window.width)}x{int(window.height)} pixels"
        )

    def refresh(self):
        """Refresh the displayed image using the current viewer state."""
        self._update_image()

    def set_relief_shading(self, enabled: bool):
        """Enable or disable relief shading and refresh the display."""
        self.show_relief = bool(enabled)
        self.refresh()

    def set_relief_intensity(self, intensity: float):
        """Set relief intensity and refresh the display if relief shading is enabled."""
        self.relief_intensity = float(intensity)
        if self.show_relief:
            self.refresh()

    def pan(self, dx_frac, dy_frac):
        view_w = self.width / self.zoom
        self.cx = np.clip(self.cx + dx_frac * view_w, 0, self.width)
        self.cy = np.clip(self.cy + dy_frac * view_w, 0, self.height)
        self.refresh()

    def zoom_at(self, factor):
        self.zoom = np.clip(self.zoom * factor, 1.0, max(1.0, self.width / 16.0))
        self.refresh()

    def show(self):
        # Create control buttons
        self.btn_up = widgets.Button(description='↑', layout=widgets.Layout(width='40px'))
        self.btn_down = widgets.Button(description='↓', layout=widgets.Layout(width='40px'))
        self.btn_left = widgets.Button(description='←', layout=widgets.Layout(width='40px'))
        self.btn_right = widgets.Button(description='→', layout=widgets.Layout(width='40px'))

        self.btn_zoom_in = widgets.Button(description='Zoom In', layout=widgets.Layout(width='80px'))
        self.btn_zoom_out = widgets.Button(description='Zoom Out', layout=widgets.Layout(width='80px'))

        self.check_relief = widgets.Checkbox(value=self.show_relief, description='Relief Shading')
        self.slider_relief = widgets.FloatSlider(
            value=self.relief_intensity,
            min=0,
            max=2.0,
            step=0.1,
            description='Relief'
        )

        # Callbacks
        self.btn_up.on_click(lambda _: self.pan(0, -0.2))
        self.btn_down.on_click(lambda _: self.pan(0, 0.2))
        self.btn_left.on_click(lambda _: self.pan(-0.2, 0))
        self.btn_right.on_click(lambda _: self.pan(0.2, 0))

        self.btn_zoom_in.on_click(lambda _: self.zoom_at(1.5))
        self.btn_zoom_out.on_click(lambda _: self.zoom_at(1 / 1.5))

        def on_relief_change(change):
            self.set_relief_shading(change['new'])
        self.check_relief.observe(on_relief_change, names='value')

        def on_relief_intensity_change(change):
            self.set_relief_intensity(change['new'])
        self.slider_relief.observe(on_relief_intensity_change, names='value')

        self.btn_stats = widgets.Button(description='Show Stats', layout=widgets.Layout(width='100px'))

        def on_stats_click(_):
            self.print_stats()
        self.btn_stats.on_click(on_stats_click)

        # Layout
        nav_controls = widgets.VBox([
            widgets.HBox([widgets.Label(layout=widgets.Layout(width='40px')), self.btn_up]),
            widgets.HBox([self.btn_left, self.btn_down, self.btn_right]),
        ])

        zoom_controls = widgets.VBox([self.btn_zoom_in, self.btn_zoom_out])

        top_row = widgets.HBox([
            nav_controls,
            widgets.Label(layout=widgets.Layout(width='20px')),
            zoom_controls,
            widgets.VBox([self.check_relief, self.slider_relief]),
            widgets.Label(layout=widgets.Layout(width='20px')),
            self.btn_stats,
        ])

        display(widgets.VBox([top_row, self.status_label, self.img_widget]))

        self.refresh()

    def print_stats(self):
        source = self.path if self.ds is not None else self.data
        stats = calculate_stats(source)
        if stats is None:
            name = self.path if self.path is not None else "array"
            print(f"No valid data found in {name}.")
            return

        name = self.path if self.path is not None else "NumPy Array"
        print(f"Statistics for {name}:")
        print(f"  Min:    {stats['min']:.2f}")
        print(f"  Max:    {stats['max']:.2f}")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std:    {stats['std']:.2f}")
        print("\nPercentiles:")
        for p, val in sorted(stats['percentiles'].items()):
            print(f"  {p:3d}%: {val:8.2f}")

        hist, bin_edges = stats['histogram']
        print("\nHistogram:")
        max_h = np.max(hist)
        for i in range(len(hist)):
            bar = "*" * int(hist[i] / max_h * 40) if max_h > 0 else ""
            print(f"  [{bin_edges[i]:8.1f}, {bin_edges[i+1]:8.1f}]: {bar} ({hist[i]})")

    def close(self):
        if hasattr(self, 'ds') and self.ds is not None:
            try:
                self.ds.close()
            except Exception:
                pass
            self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _colorize(self, data: np.ndarray) -> np.ndarray:
        """Convert elevation data to RGB using the viewer colormap."""
        if isinstance(self.cmap, str):
            cm = plt.get_cmap(self.cmap)
        else:
            cm = self.cmap

        vmin, vmax = self.vmin, self.vmax
        if not np.isfinite(vmin):
            vmin = 0
        if not np.isfinite(vmax):
            vmax = 1
        if vmax == vmin:
            vmax = vmin + 1

        norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        norm_data = np.nan_to_num(norm_data, nan=0.0)
        return cm(norm_data)[:, :, :3].astype(np.float32)


# Aliases
Viewer = MapViewer
ArrayViewer = MapViewer
from oswald.globe_viewer import GlobeViewer, globe_viewer
Globe = GlobeViewer
