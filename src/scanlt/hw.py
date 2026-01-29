from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    os: str
    machine: str
    processor: str
    is_mac: bool
    is_windows: bool
    is_linux: bool
    is_apple_silicon: bool
    env_force_backend: str | None


def get_hardware_info() -> HardwareInfo:
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    processor = platform.processor().lower()

    is_mac = sysname == "darwin"
    is_windows = sysname == "windows"
    is_linux = sysname == "linux"

    is_apple_silicon = is_mac and (machine in {"arm64", "aarch64"} or "arm" in machine)

    env_force_backend = os.environ.get("SCAN3D_BACKEND")

    return HardwareInfo(
        os=sysname,
        machine=machine,
        processor=processor,
        is_mac=is_mac,
        is_windows=is_windows,
        is_linux=is_linux,
        is_apple_silicon=is_apple_silicon,
        env_force_backend=env_force_backend,
    )
