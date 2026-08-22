import ctypes
import os
import site
import sys
from pathlib import Path

# Pre-load NVIDIA cuDNN and CUDA shared libraries into the global symbol table (RTLD_GLOBAL).
# In cuDNN 9 / CUDA 12/13 pip packages, sibling libraries (e.g., libcudnn_ops.so.9) need to be
# resolved by the dynamic linker when libcudnn.so.9 loads symbols like cudnnGetVersion.
# When running in Jupyter kernels or environments where LD_LIBRARY_PATH is not pre-exported,
# pre-loading them prevents SIGABRT ("Invalid handle. Cannot load symbol cudnnGetVersion").
def _preload_nvidia_libraries():
    search_dirs = []
    
    # 1. Check site-packages
    try:
        site_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())
        search_dirs.extend(site_dirs)
    except Exception:
        pass

    # 2. Check sys.prefix and conda prefix
    if sys.prefix:
        search_dirs.append(os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix != sys.prefix:
        search_dirs.append(os.path.join(conda_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))

    # Packages to search for shared libraries
    nvidia_packages = ["cudnn", "cublas", "cuda_runtime", "cufft", "curand", "cusolver", "cusparse"]

    loaded = set()
    for s_dir in search_dirs:
        for pkg in nvidia_packages:
            lib_dir = os.path.join(s_dir, "nvidia", pkg, "lib")
            if os.path.isdir(lib_dir):
                # Sort to ensure dependencies / base libs load cleanly
                for fname in sorted(os.listdir(lib_dir)):
                    if fname.endswith(".so") or ".so." in fname:
                        full_path = os.path.join(lib_dir, fname)
                        if full_path not in loaded and os.path.isfile(full_path):
                            try:
                                ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                                loaded.add(full_path)
                            except Exception:
                                pass

_preload_nvidia_libraries()
