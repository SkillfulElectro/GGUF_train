"""Tests for build_native.py: llama.cpp revision pinning, source discovery, generated headers."""

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "build_native.py"

ENV_VARS = (
    "GGUF_TRAIN_LLAMA_COMMIT",
    "GGUF_TRAIN_LLAMA_BRANCH",
    "GGUF_TRAIN_LLAMA_REPO",
    "GGUF_TRAIN_LLAMA_REFRESH",
)

OTHER_SHA = "0123456789abcdef0123456789abcdef01234567"


def _load_builder():
    spec = importlib.util.spec_from_file_location("_gguf_train_build_native_under_test", str(BUILD_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_tree(root: Path, llama_version=(1, 2, 3), ggml_version=(4, 5, 6), templates=True) -> Path:
    """Lay out the parts of a llama.cpp checkout the builder looks at."""
    _touch(
        root / "CMakeLists.txt",
        "project(llama)\n"
        + "".join(f"set(LLAMA_VERSION_{n} {v})\n" for n, v in zip(("MAJOR", "MINOR", "PATCH"), llama_version)),
    )
    _touch(
        root / "ggml" / "CMakeLists.txt",
        "project(ggml)\n"
        + "".join(f"set(GGML_VERSION_{n}   {v})\n" for n, v in zip(("MAJOR", "MINOR", "PATCH"), ggml_version)),
    )
    for rel in (
        # libllama
        "src/llama.cpp",
        "src/llama-context.cpp",
        "src/models/llama.cpp",
        "src/models/gemma.cpp",
        # ggml core
        "ggml/src/ggml.c",
        "ggml/src/gguf.cpp",
        # CPU backend
        "ggml/src/ggml-cpu/ggml-cpu.c",
        "ggml/src/ggml-cpu/ops.cpp",
        "ggml/src/ggml-cpu/amx/amx.cpp",
        "ggml/src/ggml-cpu/tiled/tiled.cpp",
        "ggml/src/ggml-cpu/arch/x86/quants.c",
        "ggml/src/ggml-cpu/arch/x86/repack.cpp",
        "ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp",
        "ggml/src/ggml-cpu/arch/arm/quants.c",
        "ggml/src/ggml-cpu/arch/arm/repack.cpp",
        "ggml/src/ggml-cpu/arch/powerpc/quants.c",
        # must never be compiled by the direct CPU build
        "ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp",
        "ggml/src/ggml-cuda/ggml-cuda.cpp",
        "ggml/src/ggml.h",
        "src/llama-impl.h",
        "tools/main/main.cpp",
    ):
        _touch(root / rel)
    if templates:
        _touch(
            root / "ggml" / "src" / "ggml-version.h.in",
            '#pragma once\n\n#define GGML_VERSION "@GGML_VERSION@"\n#define GGML_COMMIT  "@GGML_BUILD_COMMIT@"\n',
        )
        _touch(
            root / "src" / "llama-version.h.in",
            '#pragma once\n\n#define LLAMA_VERSION "@LLAMA_VERSION@"\n#define LLAMA_COMMIT  "@LLAMA_BUILD_COMMIT@"\n',
        )
    return root


@unittest.skipUnless(BUILD_SCRIPT.is_file(), "build_native.py is only part of the source tree")
class BuilderTestCase(unittest.TestCase):
    def setUp(self):
        self.bn = _load_builder()
        self.tmp = Path(tempfile.mkdtemp(prefix="gguf_train_build_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ENV_VARS:
            os.environ.pop(name, None)

        # Point every path the builder touches into the scratch directory.
        self.llama_dir = self.tmp / "third_party" / "llama.cpp"
        self.build_dir = self.tmp / "build" / "native"
        self.out_dir = self.tmp / "gguf_train" / "_libs"
        for name, value in (
            ("ROOT", self.tmp),
            ("LLAMA_DIR", self.llama_dir),
            ("GGML_DIR", self.llama_dir / "ggml"),
            ("BUILD_DIR", self.build_dir),
            ("BUILD_STAMP", self.build_dir / "llama_source.json"),
            ("OUT_DIR", self.out_dir),
        ):
            patcher = mock.patch.object(self.bn, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # No test may reach the network: fetches are served by a recording fake.
        self.fetches = []
        self.fetch_commit = None  # commit the fake archive claims to contain (default: the ref)
        self.fetch_error = None
        patcher = mock.patch.object(self.bn, "_fetch_llama_archive", self._fake_fetch)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.log = io.StringIO()
        patcher = mock.patch.object(self.bn, "_log", lambda msg: self.log.write(msg + "\n"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_fetch(self, repo, ref):
        self.fetches.append((repo, ref))
        if self.fetch_error is not None:
            raise self.fetch_error
        top = _make_tree(self.build_dir / "llama-download" / f"llama.cpp-{ref}")
        _touch(top / "FETCHED_REF", ref)
        commit = self.fetch_commit if self.fetch_commit is not None else (ref if len(ref) == 40 else OTHER_SHA)
        return top, commit

    def _stamp(self):
        return json.loads((self.llama_dir / self.bn.SOURCE_STAMP_NAME).read_text())


class TestRevisionSelection(BuilderTestCase):
    def test_default_revision_is_an_immutable_pinned_commit(self):
        bn = self.bn
        self.assertRegex(bn.LLAMA_PINNED_COMMIT, r"^[0-9a-f]{40}$")
        self.assertEqual(bn.llama_ref(), bn.LLAMA_PINNED_COMMIT)
        self.assertEqual(bn.llama_repo(), "https://github.com/ggml-org/llama.cpp")

    def test_environment_overrides(self):
        bn = self.bn
        os.environ["GGUF_TRAIN_LLAMA_BRANCH"] = "master"
        self.assertEqual(bn.llama_ref(), "master")
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = OTHER_SHA
        self.assertEqual(bn.llama_ref(), OTHER_SHA, "an explicit commit wins over a branch")
        os.environ["GGUF_TRAIN_LLAMA_REPO"] = "https://example.com/mirror/llama.cpp.git/"
        self.assertEqual(bn.llama_repo(), "https://example.com/mirror/llama.cpp")
        # Blank values mean "not set" rather than an empty revision.
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = "  "
        os.environ["GGUF_TRAIN_LLAMA_BRANCH"] = ""
        self.assertEqual(bn.llama_ref(), bn.LLAMA_PINNED_COMMIT)


class TestEnsureLlamaSource(BuilderTestCase):
    def test_missing_tree_fetches_the_pinned_commit_and_stamps_it(self):
        bn = self.bn
        self.assertEqual(bn.ensure_llama_source(), self.llama_dir)
        self.assertEqual(self.fetches, [("https://github.com/ggml-org/llama.cpp", bn.LLAMA_PINNED_COMMIT)])
        self.assertEqual(
            self._stamp(),
            {"repo": bn.llama_repo(), "ref": bn.LLAMA_PINNED_COMMIT, "commit": bn.LLAMA_PINNED_COMMIT},
        )
        self.assertFalse((self.build_dir / "llama-download").exists(), "download scratch dir must be cleaned up")

        # A second build reuses the tree: no network access needed.
        bn.ensure_llama_source()
        self.assertEqual(len(self.fetches), 1)

    def test_requesting_another_revision_replaces_a_stamped_tree(self):
        bn = self.bn
        bn.ensure_llama_source()
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = OTHER_SHA
        bn.ensure_llama_source()
        self.assertEqual(self.fetches[-1][1], OTHER_SHA)
        self.assertEqual(self._stamp()["commit"], OTHER_SHA)
        self.assertEqual((self.llama_dir / "FETCHED_REF").read_text(), OTHER_SHA)

        # An abbreviated sha of the commit already on disk is not a different revision.
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = OTHER_SHA[:10]
        bn.ensure_llama_source()
        self.assertEqual(len(self.fetches), 2)

        # Dropping the override returns to the pin.
        del os.environ["GGUF_TRAIN_LLAMA_COMMIT"]
        bn.ensure_llama_source()
        self.assertEqual(self.fetches[-1][1], bn.LLAMA_PINNED_COMMIT)

    def test_branch_snapshot_is_kept_until_refreshed(self):
        bn = self.bn
        os.environ["GGUF_TRAIN_LLAMA_BRANCH"] = "master"
        bn.ensure_llama_source()
        bn.ensure_llama_source()
        self.assertEqual(self.fetches, [(bn.llama_repo(), "master")])
        self.assertEqual(self._stamp(), {"repo": bn.llama_repo(), "ref": "master", "commit": OTHER_SHA})
        bn.ensure_llama_source(force_refresh=True)
        self.assertEqual(len(self.fetches), 2)

    def test_unstamped_tree_is_used_as_is_unless_asked_otherwise(self):
        bn = self.bn
        _make_tree(self.llama_dir)
        _touch(self.llama_dir / "LOCAL_PATCH", "mine")

        bn.ensure_llama_source()
        self.assertEqual(self.fetches, [], "a hand-vendored tree must not be replaced behind the user's back")
        self.assertTrue((self.llama_dir / "LOCAL_PATCH").exists())
        self.assertIn("not pinned", self.log.getvalue())

        bn.ensure_llama_source(force_refresh=True)
        self.assertEqual(self.fetches, [(bn.llama_repo(), bn.LLAMA_PINNED_COMMIT)])
        self.assertFalse((self.llama_dir / "LOCAL_PATCH").exists())

    def test_unstamped_tree_is_replaced_when_a_revision_is_requested_explicitly(self):
        bn = self.bn
        _make_tree(self.llama_dir)
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = OTHER_SHA
        bn.ensure_llama_source()
        self.assertEqual(self.fetches, [(bn.llama_repo(), OTHER_SHA)])

    def test_refresh_env_var_forces_a_download(self):
        bn = self.bn
        bn.ensure_llama_source()
        os.environ["GGUF_TRAIN_LLAMA_REFRESH"] = "1"
        bn.ensure_llama_source()
        self.assertEqual(len(self.fetches), 2)

    def test_failed_download_keeps_the_existing_tree(self):
        bn = self.bn
        bn.ensure_llama_source()
        self.fetch_error = RuntimeError("network down")
        with self.assertRaisesRegex(RuntimeError, "network down"):
            bn.ensure_llama_source(force_refresh=True)
        self.assertTrue(bn._has_llama_tree())
        self.assertEqual(self._stamp()["commit"], bn.LLAMA_PINNED_COMMIT)

    def test_archive_holding_another_commit_is_rejected(self):
        bn = self.bn
        self.fetch_commit = OTHER_SHA
        with self.assertRaisesRegex(RuntimeError, "archive contains"):
            bn.ensure_llama_source()
        self.assertFalse(self.llama_dir.exists())

    def test_git_checkout_is_never_overwritten(self):
        bn = self.bn
        _make_tree(self.llama_dir)
        (self.llama_dir / ".git").mkdir()
        bn.ensure_llama_source()  # as-is
        with self.assertRaisesRegex(RuntimeError, "git checkout"):
            bn.ensure_llama_source(force_refresh=True)
        self.assertEqual(self.fetches, [])
        self.assertTrue((self.llama_dir / ".git").is_dir())

    def test_refresh_flag_reaches_the_source_fetch(self):
        bn = self.bn

        class Reached(Exception):
            pass

        seen = {}

        def fake_ensure(force_refresh=False):
            seen["force_refresh"] = force_refresh
            raise Reached()

        with mock.patch.object(bn, "ensure_llama_source", fake_ensure), mock.patch.object(
            bn, "_find_compilers", lambda: ("cc", "c++")
        ):
            with mock.patch.object(sys, "argv", ["build_native.py", "--refresh-llama"]):
                with self.assertRaises(Reached):
                    bn.main()
            self.assertIs(seen["force_refresh"], True)

            with mock.patch.object(sys, "argv", ["build_native.py"]):
                with self.assertRaises(Reached):
                    bn.main()
            self.assertIs(seen["force_refresh"], False)


class TestSourceDiscovery(BuilderTestCase):
    def test_sources_come_from_the_tree_not_from_a_hardcoded_list(self):
        bn = self.bn
        tree = _make_tree(self.llama_dir)
        core, models = bn.collect_llama_sources(tree)
        self.assertEqual(core, ["src/llama-context.cpp", "src/llama.cpp"])
        self.assertEqual(models, ["src/models/gemma.cpp", "src/models/llama.cpp"])

        # Files upstream adds, renames or splits are picked up without touching the builder.
        _touch(tree / "src" / "llama-brand-new.cpp")
        _touch(tree / "src" / "models" / "brand-new-arch.cpp")
        (tree / "src" / "llama-context.cpp").unlink()
        _touch(tree / "ggml" / "src" / "ggml-brand-new.cpp")
        _touch(tree / "ggml" / "src" / "ggml-cpu" / "amx" / "extra.cpp")
        core, models = bn.collect_llama_sources(tree)
        self.assertEqual(core, ["src/llama-brand-new.cpp", "src/llama.cpp"])
        self.assertIn("src/models/brand-new-arch.cpp", models)
        ggml = bn.collect_ggml_sources(tree, machine="x86_64")
        self.assertIn("ggml/src/ggml-brand-new.cpp", ggml)
        self.assertIn("ggml/src/ggml-cpu/amx/extra.cpp", ggml)

    def test_ggml_sources_cover_core_and_cpu_backend_only(self):
        bn = self.bn
        tree = _make_tree(self.llama_dir)
        x86 = bn.collect_ggml_sources(tree, machine="x86_64")
        self.assertEqual(
            sorted(x86),
            sorted(
                [
                    "ggml/src/ggml.c",
                    "ggml/src/gguf.cpp",
                    "ggml/src/ggml-cpu/ggml-cpu.c",
                    "ggml/src/ggml-cpu/ops.cpp",
                    "ggml/src/ggml-cpu/amx/amx.cpp",
                    "ggml/src/ggml-cpu/tiled/tiled.cpp",
                    "ggml/src/ggml-cpu/arch/x86/quants.c",
                    "ggml/src/ggml-cpu/arch/x86/repack.cpp",
                ]
            ),
        )
        self.assertEqual(len(x86), len(set(x86)))

        def arch_files(machine):
            return sorted(s.split("ggml-cpu/arch/")[1] for s in bn.collect_ggml_sources(tree, machine=machine) if "/arch/" in s)

        self.assertEqual(arch_files("AMD64"), ["x86/quants.c", "x86/repack.cpp"])
        self.assertEqual(arch_files("aarch64"), ["arm/quants.c", "arm/repack.cpp"])
        # Only files that exist are listed (upstream has no powerpc repack.cpp).
        self.assertEqual(arch_files("ppc64le"), ["powerpc/quants.c"])
        self.assertEqual(arch_files("some-unknown-cpu"), [])

    def test_real_tree_has_every_discovered_source(self):
        real = ROOT / "third_party" / "llama.cpp"
        if not (real / "ggml" / "src" / "ggml.c").is_file():
            self.skipTest("llama.cpp sources have not been fetched")
        bn = self.bn
        ggml = bn.collect_ggml_sources(real)
        core, models = bn.collect_llama_sources(real)
        for required in ("ggml/src/ggml.c", "ggml/src/gguf.cpp", "ggml/src/ggml-opt.cpp", "ggml/src/ggml-cpu/ggml-cpu.c"):
            self.assertIn(required, ggml)
        self.assertIn("src/llama.cpp", core)
        self.assertIn("src/llama-context.cpp", core)
        self.assertGreater(len(models), 0)
        for rel in ggml + core + models:
            self.assertTrue((real / rel).is_file(), rel)


class TestGeneratedHeaders(BuilderTestCase):
    def _headers(self, gen):
        return (gen / "ggml-version.h").read_text(), (gen / "llama-version.h").read_text()

    def test_version_headers_describe_the_tree_being_built(self):
        bn = self.bn
        tree = _make_tree(self.llama_dir)
        _touch(tree / bn.SOURCE_STAMP_NAME, json.dumps({"repo": "r", "ref": "master", "commit": OTHER_SHA}))
        gen = self.build_dir / "include"
        bn._ensure_generated_headers(gen, tree)
        ggml_h, llama_h = self._headers(gen)
        self.assertIn('#define GGML_VERSION "4.5.6"', ggml_h)
        self.assertIn(f'#define GGML_COMMIT  "{OTHER_SHA}"', ggml_h)
        self.assertIn('#define LLAMA_VERSION "1.2.3"', llama_h)
        self.assertIn(f'#define LLAMA_COMMIT  "{OTHER_SHA}"', llama_h)
        self.assertNotIn("@", ggml_h + llama_h)

    def test_version_headers_are_regenerated_when_the_tree_changes(self):
        bn = self.bn
        gen = self.build_dir / "include"
        bn._ensure_generated_headers(gen, _make_tree(self.llama_dir, ggml_version=(0, 29, 0)))
        self.assertIn('"0.29.0"', self._headers(gen)[0])
        before = (gen / "llama-version.h").stat().st_mtime_ns

        shutil.rmtree(self.llama_dir)
        bn._ensure_generated_headers(gen, _make_tree(self.llama_dir, ggml_version=(0, 31, 7)))
        ggml_h, _ = self._headers(gen)
        self.assertIn('#define GGML_VERSION "0.31.7"', ggml_h)
        self.assertNotIn("0.29.0", ggml_h)
        # ...while an unchanged header is left alone, so nothing is recompiled needlessly.
        self.assertEqual((gen / "llama-version.h").stat().st_mtime_ns, before)

    def test_version_headers_without_upstream_templates(self):
        bn = self.bn
        gen = self.build_dir / "include"
        bn._ensure_generated_headers(gen, _make_tree(self.llama_dir, templates=False))
        ggml_h, llama_h = self._headers(gen)
        self.assertIn('#define GGML_VERSION "4.5.6"', ggml_h)
        self.assertIn('#define GGML_COMMIT "unknown"', ggml_h)
        self.assertIn('#define LLAMA_VERSION "1.2.3"', llama_h)


class TestBuildStamp(BuilderTestCase):
    def _populate_build_products(self):
        bn = self.bn
        _touch(self.build_dir / "obj" / "ggml" / "src" / "ggml.c.o", "stale")
        _touch(self.build_dir / "gen_src" / "llama_models_unity_00.cpp", "stale")
        _touch(self.build_dir / "include" / "ggml-version.h", "stale")
        libs = [_touch(self.out_dir / bn.lib_filename(stem), "stale") for stem in ("ggml_train_ggml", "gguf_train", "llama_train_llama", "llama_bridge")]
        return libs

    def test_products_of_another_revision_are_discarded(self):
        bn = self.bn
        bn.ensure_llama_source()
        libs = self._populate_build_products()

        self.assertTrue(bn._sync_build_stamp())
        self.assertFalse((self.build_dir / "obj").exists())
        self.assertFalse((self.build_dir / "gen_src").exists())
        self.assertFalse(any(lib.exists() for lib in libs))
        self.assertEqual(json.loads(bn.BUILD_STAMP.read_text()), self._stamp())

        # Same revision again: an interrupted or finished build is resumed, not restarted.
        libs = self._populate_build_products()
        self.assertFalse(bn._sync_build_stamp())
        self.assertTrue((self.build_dir / "obj" / "ggml" / "src" / "ggml.c.o").exists())
        self.assertTrue(all(lib.exists() for lib in libs))

        # Switching revision invalidates everything again.
        os.environ["GGUF_TRAIN_LLAMA_COMMIT"] = OTHER_SHA
        bn.ensure_llama_source()
        self.assertTrue(bn._sync_build_stamp())
        self.assertFalse(any(lib.exists() for lib in libs))
        self.assertEqual(json.loads(bn.BUILD_STAMP.read_text())["commit"], OTHER_SHA)

    def test_unmanaged_tree_keeps_relying_on_timestamps(self):
        bn = self.bn
        _make_tree(self.llama_dir)
        libs = self._populate_build_products()
        self.assertFalse(bn._sync_build_stamp())
        self.assertTrue(all(lib.exists() for lib in libs))
        self.assertFalse(bn.BUILD_STAMP.exists())


class TestCompileUnit(BuilderTestCase):
    def test_failed_compile_leaves_no_object_behind(self):
        bn = self.bn
        src = _touch(self.tmp / "unit.c", "int f(void) { return 0; }\n")
        obj = self.tmp / "obj" / "unit.c.o"
        failing = [sys.executable, "-c", "import sys; open(sys.argv[-1], 'w').write('partial'); sys.exit(3)"]
        with self.assertRaisesRegex(RuntimeError, "Compilation failed"):
            bn._compile_unit(failing, src, obj, force=True)
        self.assertEqual(list(obj.parent.iterdir()), [], "a failed compile must not leave a partial object")

        ok = [sys.executable, "-c", "import sys; open(sys.argv[-1], 'w').write('object code')"]
        self.assertEqual(bn._compile_unit(ok, src, obj, force=True), obj)
        self.assertEqual(obj.read_text(), "object code")
        self.assertEqual([p.name for p in obj.parent.iterdir()], ["unit.c.o"])


if __name__ == "__main__":
    unittest.main()
