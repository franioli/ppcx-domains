import ctypes
import glob
import os
import site


def _setup_nvidia_libs():
    """Preload NVIDIA shared libraries at Python startup."""
    site_packages = site.getsitepackages()[0]
    nvidia_lib_dirs = glob.glob(os.path.join(site_packages, "nvidia", "*", "lib"))

    if not nvidia_lib_dirs:
        return

    nvidia_path = ":".join(nvidia_lib_dirs)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{nvidia_path}:{existing}" if existing else nvidia_path
    )

    for lib_dir in nvidia_lib_dirs:
        for so_file in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
            try:
                ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_setup_nvidia_libs()
