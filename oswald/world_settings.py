import numpy as np
from PIL import Image


def set_sea_level(source_image, sea_level):
    arr = np.array(source_image, dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        elif arr.shape[2] == 3:
            channel_spread = arr.max(axis=2) - arr.min(axis=2)
            if np.max(channel_spread) > 1.0:
                raise ValueError("source_image must represent a single-channel height field")
            arr = arr.mean(axis=2)
        else:
            raise ValueError("source_image must represent a single-channel height field")

    if arr.size == 0:
        return source_image.copy()

    if not 0.0 <= sea_level <= 1.0:
        raise ValueError("sea_level must be between 0 and 1")

    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        normalized = (arr - arr_min) / (arr_max - arr_min)

    if sea_level == 0.0:
        adjusted = 0.5 + 0.5 * normalized
    elif sea_level == 1.0:
        adjusted = 0.5 * normalized
    else:
        adjusted = np.empty_like(normalized, dtype=np.float32)
        below = normalized <= sea_level
        adjusted[below] = 0.5 * (normalized[below] / sea_level)
        adjusted[~below] = 0.5 + 0.5 * ((normalized[~below] - sea_level) / (1.0 - sea_level))

    adjusted = np.clip(adjusted, 0.0, 1.0)
    scaled = np.round(adjusted * np.iinfo(np.uint16).max).astype(np.uint16)
    return Image.fromarray(scaled, mode="I;16")
