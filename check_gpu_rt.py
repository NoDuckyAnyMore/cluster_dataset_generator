"""Minimal CUDA/OptiX smoke test for the Sionna RT Ubuntu environment."""

from __future__ import annotations

import ctypes.util
import os
from importlib.metadata import PackageNotFoundError, version

from run_cluster_5090 import configure_optix_library


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    configure_optix_library()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"DRJIT_LIBOPTIX_PATH={os.environ.get('DRJIT_LIBOPTIX_PATH', '<not set>')}")
    print(f"OPTIX_CACHE_PATH={os.environ.get('OPTIX_CACHE_PATH', '<not set>')}")
    print(f"CUDA_CACHE_PATH={os.environ.get('CUDA_CACHE_PATH', '<not set>')}")
    print(f"loader nvoptix={ctypes.util.find_library('nvoptix') or '<not found>'}")
    print(
        "versions: "
        f"sionna-rt={package_version('sionna-rt')}, "
        f"mitsuba={package_version('mitsuba')}, "
        f"drjit={package_version('drjit')}"
    )

    import sionna.rt  # noqa: F401 - configures Sionna's Mitsuba variant
    import mitsuba as mi

    print(f"Mitsuba variant={mi.variant()}")
    scene = mi.load_dict({"type": "scene"})
    print(f"OptiX scene creation OK: {type(scene).__name__}")


if __name__ == "__main__":
    main()
