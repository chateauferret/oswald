import ctypes
import os
import site
import sys


def configure_torch_cuda_allocator():
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def preload_nvidia_libraries():
    search_dirs = []

    try:
        site_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())
        search_dirs.extend(site_dirs)
    except Exception:
        pass

    if sys.prefix:
        search_dirs.append(
            os.path.join(
                sys.prefix,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
        )

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix != sys.prefix:
        search_dirs.append(
            os.path.join(
                conda_prefix,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
        )

    nvidia_packages = [
        "cudnn",
        "cublas",
        "cuda_runtime",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
    ]

    loaded = set()
    for site_dir in search_dirs:
        for package in nvidia_packages:
            lib_dir = os.path.join(site_dir, "nvidia", package, "lib")
            if not os.path.isdir(lib_dir):
                continue

            for filename in sorted(os.listdir(lib_dir)):
                if not (filename.endswith(".so") or ".so." in filename):
                    continue

                full_path = os.path.join(lib_dir, filename)
                if full_path in loaded or not os.path.isfile(full_path):
                    continue

                try:
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                    loaded.add(full_path)
                except Exception:
                    pass
