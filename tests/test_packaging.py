"""Packaging checks.

The wheel must carry the compiled libraries and be tagged for the platform it was built on,
the sdist must carry sources only, and an installed copy must import without the source tree.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "gguf_train"
LIBS_DIR = PKG_DIR / "_libs"
BINARY_SUFFIXES = (".so", ".dylib", ".dll", ".o", ".obj", ".a", ".lib", ".pyc")


def _lib_name(stem: str) -> str:
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    if sys.platform == "win32":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _built_libs() -> list:
    stems = ("ggml_train_ggml", "gguf_train", "llama_train_llama", "llama_bridge")
    return [_lib_name(s) for s in stems if (LIBS_DIR / _lib_name(s)).is_file()]


def _has_bdist_wheel() -> bool:
    for name in ("setuptools.command.bdist_wheel", "wheel.bdist_wheel"):
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except ImportError:
            continue
    return False


def _clean_env(**extra) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # Packaging is what is under test here, not compilation: reuse the libraries already built.
    env["GGUF_TRAIN_SKIP_BUILD"] = "1"
    env.update(extra)
    return env


def _run_setup(args, tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "setup.py", "-q", *args],
        cwd=str(ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_isolated(code: str, site: Path, cwd: Path) -> subprocess.CompletedProcess:
    """Run `code` with only `site` (not the source tree) providing gguf_train/torch."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=_clean_env(PYTHONPATH=str(site)),
        capture_output=True,
        text=True,
        timeout=600,
    )


@unittest.skipUnless((ROOT / "setup.py").is_file(), "packaging is tested from the source tree")
@unittest.skipUnless(importlib.util.find_spec("setuptools") is not None, "setuptools is not installed")
class PackagingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gguf_train_pkg_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class TestWheel(PackagingTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._wheel_dir = None
        cls.wheel = None
        cls.build_log = ""
        if not _has_bdist_wheel() or not (LIBS_DIR / _lib_name("gguf_train")).is_file():
            return
        cls._wheel_dir = Path(tempfile.mkdtemp(prefix="gguf_train_wheel_test_"))
        tmp = cls._wheel_dir
        # Every intermediate goes to the scratch directory so the source tree stays clean.
        res = _run_setup(
            [
                "build", "--build-base", str(tmp / "build"),
                "egg_info", "--egg-base", str(tmp),
                "bdist_wheel", "--dist-dir", str(tmp / "dist"), "--bdist-dir", str(tmp / "bdist"),
            ],
            tmp,
        )
        cls.build_log = res.stdout + res.stderr
        wheels = sorted((tmp / "dist").glob("*.whl")) if res.returncode == 0 else []
        cls.wheel = wheels[0] if len(wheels) == 1 else None

    @classmethod
    def tearDownClass(cls):
        if cls._wheel_dir is not None:
            shutil.rmtree(cls._wheel_dir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        if not _has_bdist_wheel():
            self.skipTest("bdist_wheel is not available (install `wheel` or setuptools>=70.1)")
        if not (LIBS_DIR / _lib_name("gguf_train")).is_file():
            self.skipTest("native libraries have not been built")
        if self.wheel is None:
            self.fail("building the wheel failed:\n" + self.build_log[-4000:])

    def test_wheel_is_tagged_for_this_platform(self):
        # gguf_train-<version>-<python>-<abi>-<platform>.whl
        python_tag, abi_tag, platform_tag = self.wheel.name[: -len(".whl")].split("-")[-3:]
        self.assertNotEqual(platform_tag, "any", "a wheel holding native code must not claim to run anywhere")
        # ctypes-loaded libraries do not depend on the interpreter version or ABI.
        self.assertEqual((python_tag, abi_tag), ("py3", "none"))

        with zipfile.ZipFile(self.wheel) as z:
            meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/WHEEL"))
            meta = z.read(meta_name).decode()
        self.assertIn("Root-Is-Purelib: false", meta)
        self.assertIn(f"Tag: py3-none-{platform_tag}", meta)

    def test_wheel_contains_the_native_libraries(self):
        with zipfile.ZipFile(self.wheel) as z:
            names = set(z.namelist())
        built = _built_libs()
        self.assertIn(_lib_name("gguf_train"), built)
        self.assertIn(_lib_name("ggml_train_ggml"), built)
        for lib in built:
            self.assertIn(f"gguf_train/_libs/{lib}", names)
        self.assertIn("gguf_train/__init__.py", names)
        self.assertIn("torch/__init__.py", names)
        # Build inputs and intermediates stay out of the wheel.
        self.assertFalse([n for n in names if n.endswith((".o", ".cpp", ".pyc")) or n.endswith("build_native.py")])

    def test_installed_wheel_imports_without_the_source_tree(self):
        site = self.tmp / "site"
        with zipfile.ZipFile(self.wheel) as z:
            z.extractall(site)
        has_bridge = (LIBS_DIR / _lib_name("llama_bridge")).is_file()
        code = (
            "import os, gguf_train, torch\n"
            "from gguf_train import _C\n"
            "x = torch.arange(6.0).reshape(2, 3)\n"
            "assert (x @ x.t()).tolist() == [[5.0, 14.0], [14.0, 50.0]]\n"
            + ("assert _C.llama_lib() is not None\n" if has_bridge else "")
            + "print(os.path.dirname(os.path.abspath(gguf_train.__file__)))\n"
            "print(os.path.dirname(os.path.abspath(torch.__file__)))\n"
        )
        res = _run_isolated(code, site, self.tmp)
        self.assertEqual(res.returncode, 0, res.stderr)
        locations = [Path(line).resolve() for line in res.stdout.split()]
        self.assertEqual(locations, [(site / "gguf_train").resolve(), (site / "torch").resolve()])
        self.assertNotIn("building native", res.stderr)


class TestInstalledCopyWithoutLibraries(PackagingTestCase):
    def test_missing_libraries_give_an_actionable_error_instead_of_a_build_attempt(self):
        site = self.tmp / "site"
        shutil.copytree(PKG_DIR, site / "gguf_train", ignore=shutil.ignore_patterns("_libs", "__pycache__"))
        # A stray build_native.py beside the installed package (site-packages is shared with
        # every other distribution) must never be picked up and executed.
        marker = self.tmp / "decoy_was_executed"
        (site / "build_native.py").write_text(
            f"open({str(marker)!r}, 'w').close()\n"
            "def build_direct(**kwargs):\n"
            f"    open({str(marker)!r}, 'w').close()\n"
        )
        res = _run_isolated("import gguf_train", site, self.tmp)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(marker.exists(), "an unrelated build_native.py was executed")
        self.assertIn("RuntimeError", res.stderr)
        self.assertIn(_lib_name("gguf_train"), res.stderr)
        self.assertIn("Reinstall", res.stderr)
        self.assertNotIn("building native", res.stderr)


class TestSdistManifest(PackagingTestCase):
    def test_sdist_lists_sources_but_no_compiled_artifacts(self):
        if not _built_libs():
            self.skipTest("native libraries have not been built, so there is nothing to leak into the sdist")
        res = _run_setup(["egg_info", "--egg-base", str(self.tmp)], self.tmp)
        self.assertEqual(res.returncode, 0, res.stderr)
        sources_file = next(self.tmp.glob("*.egg-info")) / "SOURCES.txt"
        sources = [line.strip().replace("\\", "/") for line in sources_file.read_text().splitlines() if line.strip()]

        # Compiled libraries are machine-specific build products; an sdist carrying them would
        # make the native build look up to date on whatever machine it is unpacked on.
        self.assertEqual([s for s in sources if s.endswith(BINARY_SUFFIXES)], [])
        self.assertEqual([s for s in sources if s.startswith("gguf_train/_libs/")], [])

        for required in (
            "setup.py",
            "pyproject.toml",
            "MANIFEST.in",
            "build_native.py",
            "csrc/gguf_train.h",
            "csrc/gt_core.cpp",
            "csrc/llama_bridge.cpp",
            "csrc/CMakeLists.txt",
            "gguf_train/__init__.py",
            "gguf_train/_C.py",
            "torch/__init__.py",
        ):
            self.assertIn(required, sources)

        # A fetched llama.cpp tree travels with the sdist together with its revision stamp,
        # which keeps a build from the sdist pinned and possible without network access.
        if (ROOT / "third_party" / "llama.cpp" / "gguf_train_source.json").is_file():
            self.assertIn("third_party/llama.cpp/gguf_train_source.json", sources)
            self.assertIn("third_party/llama.cpp/ggml/src/ggml.c", sources)


if __name__ == "__main__":
    unittest.main()
