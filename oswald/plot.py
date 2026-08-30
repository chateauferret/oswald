from pathlib import Path
from base64 import b64encode
from io import BytesIO
from uuid import uuid4
import json

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import ast
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from IPython.display import HTML, display

def load_topo_cmap():
    legend_path = Path(__file__).resolve().parent.parent / "legends" / "topography.txt"
    with legend_path.open("r") as data:
        cdict = ast.literal_eval(data.read())
    return LinearSegmentedColormap("topo", cdict)

def plot(source_image, sea_level=0):
    topo = load_topo_cmap().copy()
    topo.set_bad(alpha=0)

    source_data = np.array(source_image, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(2.5, 2.5))
    pic = axis.imshow(
        source_data,
        cmap=topo,
        interpolation='nearest',
        vmin=-32767,
        vmax=32767

    )
    axis.axis('off')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    div = make_axes_locatable(axis)
    cx = div.append_axes("right", size="5%", pad=0.05)

    legend = figure.colorbar(pic, cax=cx)
    legend.set_ticks([-32767, -16384, 0, 16384, 32767])
    legend.set_label('Height (feet)')

    var = plt.draw

    plt.show(var)


def browse(source_images, titles=None):
    if not source_images:
        raise ValueError("source_images must not be empty")

    topo = load_topo_cmap().copy()
    topo.set_bad(alpha=0)

    if titles is None:
        titles = [f"Image {index + 1}" for index in range(len(source_images))]
    elif len(titles) != len(source_images):
        raise ValueError("titles must match source_images length")

    normalizer = Normalize(vmin=-32767, vmax=32767, clip=True)
    rendered_images = []

    for source_image in source_images:
        source_data = np.array(source_image, dtype=np.float32)
        rgba = topo(normalizer(source_data), bytes=True)
        image = Image.fromarray(rgba, mode="RGBA")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        rendered_images.append(f"data:image/png;base64,{b64encode(buffer.getvalue()).decode('ascii')}")

    container_id = f"terrain-browser-{uuid4().hex}"
    image_entries = json.dumps(rendered_images)
    title_entries = json.dumps(titles)

    html = f"""
    <div id="{container_id}" style="width: min-content; text-align: center;">
      <div style="margin-bottom: 0.5rem; font-weight: 600;"></div>
      <img style="display: block; max-width: 600px; max-height: 600px; border: 1px solid #ccc;" />
      <div style="margin-top: 0.75rem;">
        <button type="button">Previous</button>
        <button type="button" style="margin-left: 0.5rem;">Next</button>
      </div>
    </div>
    <script>
    (() => {{
      const container = document.getElementById("{container_id}");
      const titles = {title_entries};
      const images = {image_entries};
      let index = 0;
      const title = container.children[0];
      const image = container.children[1];
      const previous = container.children[2].children[0];
      const next = container.children[2].children[1];

      const render = () => {{
        title.textContent = `${{titles[index]}} (${{index + 1}}/${{images.length}})`;
        image.src = images[index];
        previous.disabled = index === 0;
        next.disabled = index === images.length - 1;
      }};

      previous.addEventListener("click", () => {{
        if (index > 0) {{
          index -= 1;
          render();
        }}
      }});

      next.addEventListener("click", () => {{
        if (index < images.length - 1) {{
          index += 1;
          render();
        }}
      }});

      render();
    }})();
    </script>
    """

    browser = HTML(html)
    display(browser)
    return browser