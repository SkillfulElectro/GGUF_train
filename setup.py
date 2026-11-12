from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 ships bdist_wheel itself
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    try:
        from wheel.bdist_wheel import bdist_wheel
    except ImportError:  # no wheel support installed; sdist and plain installs still work
        bdist_wheel = None

ROOT = Path(__file__).resolve().parent
LIBS_DIR = ROOT / "gguf_train" / "_libs"
NATIVE_LIB_GLOBS = ("*.so", "*.dylib", "*.dll")

_native_build_done = False


def _has_native_libs() -> bool:
    return any(True for pattern in NATIVE_LIB_GLOBS for _ in LIBS_DIR.glob(pattern))


def _run_native_build():
    global _native_build_done
    if _native_build_done:
        return
    if os.environ.get("GGUF_TRAIN_SKIP_BUILD") == "1":
        if not _has_native_libs():
            sys.stderr.write(
                "[gguf_train] WARNING: GGUF_TRAIN_SKIP_BUILD=1 and gguf_train/_libs holds no native "
                "libraries. The package being built will not be importable once installed; run "
                "`python build_native.py` first or unset GGUF_TRAIN_SKIP_BUILD.\n"
            )
        return
    subprocess.check_call([sys.executable, str(ROOT / "build_native.py")])
    _native_build_done = True


class BinaryDistribution(Distribution):
    """gguf_train ships compiled shared libraries, so it is not a pure-Python distribution.

    Without this the wheel is tagged ``py3-none-any`` and installs into purelib, i.e. a
    Linux build claims to be installable on every platform.
    """

    def has_ext_modules(self):
        return True


class BuildPyWithNative(build_py):
    def run(self):
        _run_native_build()
        # setuptools computes the package-data list lazily and caches it on the command.
        # Anything that asked for it before the libraries were built (another command of
        # the same invocation, a plugin, ...) would have cached a list without them.
        if hasattr(self, "_get_data_files"):
            self.data_files = self._get_data_files()
        super().run()


class DevelopWithNative(develop):
    def run(self):
        _run_native_build()
        super().run()


class InstallWithNative(install):
    def run(self):
        _run_native_build()
        super().run()


cmdclass = {
    "build_py": BuildPyWithNative,
    "develop": DevelopWithNative,
    "install": InstallWithNative,
}

if bdist_wheel is not None:

    class BdistWheelNative(bdist_wheel):
        """Tag wheels ``py3-none-<platform>``.

        The native libraries are loaded through ctypes and never link against libpython,
        so a wheel is specific to the platform it was built on but works with every
        Python 3 version there (rather than only the interpreter that built it).
        """

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = BdistWheelNative


setup(
    name="gguf-train",
    version="0.1.0",
    packages=find_packages(include=["gguf_train*", "torch*"]),
    package_data={"gguf_train": ["_libs/" + pattern for pattern in NATIVE_LIB_GLOBS]},
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
)
