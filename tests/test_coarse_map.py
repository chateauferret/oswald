import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from PIL import Image

from oswald.coarse_map import CoarseMap, _extract_channels


def test_coarse_map_init_and_context_manager():
    with CoarseMap(size=64, extent=3000) as cmap:
        assert isinstance(cmap.coarse_map, dict)
        assert len(cmap.coarse_map) == 0
        assert cmap.size == 64
        assert cmap.extent == 3000.0

    arr = np.ones((64, 128), dtype=np.float32)
    cmap2 = CoarseMap(coarse_maps=arr)
    assert 0 in cmap2.coarse_map
    assert cmap2.coarse_map[0].shape == (64, 128)


def test_add_channel_and_get_cond_map_workflow():
    # Test the exact workflow described in lines 6-17 of coarse_map.py:
    # mapper = cm.CoarseMap(size=64, extent=3000)
    # mapper.add_channel(coarse_map=world_coarse_map, channel="heightmap")
    # cond_map = mapper.get_cond_map(lat=45.0, lon=45.0)
    h, w = 180, 360
    lats = np.linspace(90, -90, h)
    world_arr = np.repeat(lats[:, None], w, axis=1).astype(np.float32)
    world_coarse_map = Image.fromarray(world_arr)

    mapper = CoarseMap(size=64, extent=3000)
    mapper.add_channel(coarse_map=world_coarse_map, channel="heightmap")
    cond_map1 = mapper.get_cond_map(lat=45.0, lon=45.0)
    assert isinstance(cond_map1, np.ndarray)
    assert cond_map1.shape == (64, 64)

    # Calling repeatedly with different lat/lon
    cond_map2 = mapper.get_cond_map(lat=0.0, lon=0.0)
    assert isinstance(cond_map2, np.ndarray)
    assert cond_map2.shape == (64, 64)
    assert abs(cond_map2[32, 32] - 0.0) < 1.0

    # Add second channel (temperature) as numpy array
    temp_arr = np.ones((h, w), dtype=np.float32) * 20.0
    mapper.add_channel(coarse_map=temp_arr, channel="temperature")
    cond_stack = mapper.get_cond_map(lat=45.0, lon=45.0)
    assert isinstance(cond_stack, np.ndarray)
    assert cond_stack.shape == (2, 64, 64)
    np.testing.assert_allclose(cond_stack[1], 20.0, rtol=1e-4)


def test_load_course_map_from_array():
    cmap = CoarseMap()
    elev = np.random.randn(64, 128).astype(np.float32)
    temp = np.random.randn(64, 128).astype(np.float32)
    precip = np.random.randn(64, 128).astype(np.float32)

    cmap.load_coarse_map_from_array(elev, channel="heightmap")
    cmap.load_coarse_map_from_array(temp, channel="temperature")
    cmap.load_coarse_map_from_array(precip, channel=3)

    assert 0 in cmap.coarse_map
    assert 1 in cmap.coarse_map
    assert 3 in cmap.coarse_map
    np.testing.assert_array_equal(cmap.coarse_map[0], elev)
    np.testing.assert_array_equal(cmap.coarse_map[1], temp)
    np.testing.assert_array_equal(cmap.coarse_map[3], precip)


def test_load_coarse_map_stack_from_array():
    cmap = CoarseMap()
    stack = np.random.randn(3, 64, 128).astype(np.float32)
    cmap.load_coarse_map_stack_from_array(stack)

    assert len(cmap.coarse_map) == 3
    for c in range(3):
        assert c in cmap.coarse_map
        np.testing.assert_array_equal(cmap.coarse_map[c], stack[c])


def test_load_from_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h, w = 64, 128
        arr = (np.random.rand(h, w) * 100).astype(np.float32)

        # 1. Single channel GeoTIFF
        tif_path = tmp_path / "heightmap.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32"
        ) as dst:
            dst.write(arr, 1)

        cmap = CoarseMap()
        cmap.load_coarse_map_from_file(tif_path, channel="heightmap")
        assert 0 in cmap.coarse_map
        np.testing.assert_allclose(cmap.coarse_map[0], arr, rtol=1e-5)

        # Test reversed arg order as in docstring usage
        cmap_rev = CoarseMap()
        cmap_rev.load_coarse_map("elevation", str(tif_path))
        assert 0 in cmap_rev.coarse_map
        np.testing.assert_allclose(cmap_rev.coarse_map[0], arr, rtol=1e-5)

        # 2. Multi-band GeoTIFF stack
        stack_path = tmp_path / "stack.tif"
        stack_data = (np.random.rand(3, h, w) * 50).astype(np.float32)
        with rasterio.open(
            stack_path, "w", driver="GTiff", height=h, width=w, count=3, dtype="float32"
        ) as dst:
            dst.write(stack_data)

        cmap_stack = CoarseMap()
        cmap_stack.load_coarse_map_stack_from_file(stack_path)
        assert len(cmap_stack.coarse_map) == 3
        for c in range(3):
            np.testing.assert_allclose(cmap_stack.coarse_map[c], stack_data[c], rtol=1e-5)

        # 3. .npy and .npz
        npy_path = tmp_path / "temp.npy"
        np.save(npy_path, arr)
        cmap_npy = CoarseMap()
        cmap_npy.load_coarse_map_from_file(npy_path, channel="temperature")
        assert 1 in cmap_npy.coarse_map
        np.testing.assert_allclose(cmap_npy.coarse_map[1], arr)


def test_get_conditioning_map():
    h, w = 180, 360
    # Create an equirectangular world with known values
    # e.g., grid of latitude values
    lats = np.linspace(90, -90, h)
    lat_grid = np.repeat(lats[:, None], w, axis=1).astype(np.float32)

    cmap = CoarseMap()
    cmap.load_coarse_map_from_array(lat_grid, channel="heightmap")

    # Center at lat=45, lon=0, small extent
    cond = cmap.get_conditioning_map(lat=45.0, lon=0.0, size=64, extent=500.0)
    assert isinstance(cond, np.ndarray)
    assert cond.shape == (64, 64)
    # Center pixel should be close to 45 degrees
    center_val = np.mean(cond[31:33, 31:33])
    assert abs(center_val - 45.0) < 0.5

    # Center at South pole
    cond_sp = cmap.get_conditioning_map(lat=-90.0, lon=0.0, size=64, extent=500.0)
    assert cond_sp.shape == (64, 64)
    assert cond_sp[32, 32] < -85.0

    # Multi-channel conditioning map
    temp_grid = np.ones((h, w), dtype=np.float32) * 25.0
    cmap.load_coarse_map_from_array(temp_grid, channel="temperature")

    cond_multi = cmap.get_conditioning_map(lat=0.0, lon=0.0, size=32, extent=1000.0)
    assert isinstance(cond_multi, np.ndarray)
    assert cond_multi.shape == (2, 32, 32)
    # Channel 0 is heightmap (lat_grid at equator is ~0)
    assert abs(cond_multi[0, 16, 16]) < 1.0
    # Channel 1 is temperature
    np.testing.assert_allclose(cond_multi[1], 25.0, rtol=1e-4)


def test_out_of_bounds_default_fill():
    h, w = 64, 128
    arr = np.ones((h, w), dtype=np.float32) * 500.0

    cmap = CoarseMap()
    cmap.load_coarse_map_from_array(arr, channel="heightmap")

    # Request extent larger than Earth radius (R = 6371 km)
    cond = cmap.get_conditioning_map(lat=0.0, lon=0.0, size=128, extent=8000.0)
    # Corners (distance > 6371 km) should be filled with default_value -1000.0
    assert cond[0, 0] == -1000.0
    assert cond[0, 127] == -1000.0
    assert cond[127, 0] == -1000.0
    assert cond[127, 127] == -1000.0
    # Center is on globe
    assert cond[64, 64] == 500.0


def test_ingestion_into_extract_channels():
    # Verify that the conditioning map output can be ingested by _extract_channels (as used by MapGenerator)
    h, w = 64, 128
    arr = np.random.randn(h, w).astype(np.float32)
    cmap = CoarseMap()
    cmap.load_coarse_map_from_array(arr, channel="heightmap")

    cond_single = cmap.get_conditioning_map(lat=30.0, lon=45.0, size=64, extent=1000.0)
    extracted_single = _extract_channels(cond_single)
    assert 0 in extracted_single
    assert extracted_single[0].shape == (64, 64)

    # With multiple channels
    cmap.load_coarse_map_from_array(arr + 10, channel="temperature")
    cond_multi = cmap.get_conditioning_map(lat=30.0, lon=45.0, size=64, extent=1000.0)
    extracted_multi = _extract_channels(cond_multi)
    assert 0 in extracted_multi
    assert 1 in extracted_multi
    assert extracted_multi[0].shape == (64, 64)
    assert extracted_multi[1].shape == (64, 64)


def test_antimeridian_wrap():
    h, w = 180, 360
    # Longitude grid from -180 to 180
    lons = np.linspace(-180, 180, w, endpoint=False)
    lon_grid = np.repeat(lons[None, :], h, axis=0).astype(np.float32)

    cmap = CoarseMap()
    cmap.load_coarse_map_from_array(lon_grid, channel="heightmap")

    # Center projection at lon=180, lat=0
    cond = cmap.get_conditioning_map(lat=0.0, lon=180.0, size=64, extent=1000.0)
    assert cond.shape == (64, 64)
    # At center (lon=180 = -180), values should be near 180 or -180 without NaNs or crashes
    assert np.all(np.isfinite(cond))


def test_channel_aliases():
    cmap = CoarseMap()
    arr = np.ones((10, 20), dtype=np.float32)
    cmap.load_coarse_map_from_array(arr * 1, channel="elev")
    cmap.load_coarse_map_from_array(arr * 2, channel="temp")
    cmap.load_coarse_map_from_array(arr * 3, channel="temp_std")
    cmap.load_coarse_map_from_array(arr * 4, channel="rain")
    cmap.load_coarse_map_from_array(arr * 5, channel="precip_cv")

    assert cmap.coarse_map[0][0, 0] == 1
    assert cmap.coarse_map[1][0, 0] == 2
    assert cmap.coarse_map[2][0, 0] == 3
    assert cmap.coarse_map[3][0, 0] == 4
    assert cmap.coarse_map[4][0, 0] == 5
