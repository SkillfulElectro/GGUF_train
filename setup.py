from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

ROOT = Path(__file__).resolve().parent


def _run_native_build():
    if os.environ.get("GGUF_TRAIN_SKIP_BUILD") == "1":
        return
    subprocess.check_call([sys.executable, str(ROOT / "build_native.py")])


class BuildPyWithNative(build_py):
    def run(self):
        _run_native_build()
        super().run()


class DevelopWithNative(develop):
    def run(self):
        _run_native_build()
        super().run()


class InstallWithNative(install):
    def run(self):
        _run_native_build()
        super().run()


setup(
    name="gguf-train",
    version="0.1.0",
    packages=find_packages(include=["gguf_train*", "torch*"]),
    package_data={"gguf_train": ["_libs/*.so", "_libs/*.dylib", "_libs/*.dll"]},
    include_package_data=True,
    cmdclass={
        "build_py": BuildPyWithNative,
        "develop": DevelopWithNative,
        "install": InstallWithNative,
    },
    zip_safe=False,
)
