#!/usr/bin/env python3
"""
Dependency-free native builder for gguf_train.

Fetches a *pinned* llama.cpp revision (ggml + llama) into third_party/llama.cpp
and compiles it together with csrc/ into shared libraries under
gguf_train/_libs/ using gcc/g++ or clang/clang++ directly (no CMake or Python.h
required), with optional CMake GPU backend support.

The revision is pinned (LLAMA_PINNED_COMMIT) so that every build of a given
gguf_train checkout compiles exactly the llama.cpp sources it was developed and
tested against. Override it with:

  GGUF_TRAIN_LLAMA_COMMIT   exact commit sha (or tag) to build against
  GGUF_TRAIN_LLAMA_BRANCH   branch/tag to track instead (e.g. "master")
  GGUF_TRAIN_LLAMA_REPO     alternate upstream (mirrors/enterprise forks)
  GGUF_TRAIN_LLAMA_REFRESH  set to "1" (or pass --refresh-llama) to re-download

The list of llama.cpp/ggml translation units is discovered from the fetched tree
rather than hardcoded, so upstream adding, renaming or splitting source files
does not break the build.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
GGML_DIR = LLAMA_DIR / "ggml"
OUT_DIR = ROOT / "gguf_train" / "_libs"
BUILD_DIR = ROOT / "build" / "native"

# llama.cpp revision gguf_train is built and tested against (release tag b12283).
# A full commit sha is used rather than the tag name because a sha is immutable.
# Bump both together after re-running the test suite against the new revision.
LLAMA_PINNED_COMMIT = "f3b407ca1bfeeecebf9e9ef19fcd6eed565bd70a"
LLAMA_PINNED_TAG = "b12283"
LLAMA_DEFAULT_REPO = "https://github.com/ggml-org/llama.cpp"

# Written into the fetched tree; records which revision it holds.
SOURCE_STAMP_NAME = "gguf_train_source.json"
# Written next to the object files; records which revision they were compiled from.
BUILD_STAMP = BUILD_DIR / "llama_source.json"

# ggml-cpu sub-directories that upstream compiles unconditionally. Every other
# sub-directory (llamafile, kleidiai, spacemit, ...) is an opt-in feature that
# needs extra flags/dependencies, and arch/<name> is selected per target CPU.
GGML_CPU_SUBDIRS = ("amx", "tiled")

# platform.machine() -> ggml-cpu/arch/<name>
_ARCH_DIRS = (
    (("x86_64", "amd64", "i686", "i386", "x86"), "x86"),
    (("arm64", "aarch64", "armv7l", "armv8l"), "arm"),
    (("ppc64", "ppc64le", "powerpc64", "powerpc64le"), "powerpc"),
    (("riscv64", "riscv32"), "riscv"),
    (("s390x", "s390"), "s390"),
    (("loongarch64",), "loongarch"),
)
# Only these per-arch files belong to the statically linked CPU backend;
# cpu-feats.cpp is exclusively for upstream's dlopen()ed backend variants.
_ARCH_FILES = ("quants.c", "repack.cpp")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def llama_repo() -> str:
    repo = _env("GGUF_TRAIN_LLAMA_REPO", LLAMA_DEFAULT_REPO).rstrip("/")
    return repo[:-4] if repo.endswith(".git") else repo


def llama_ref() -> str:
    """Revision to build against: explicit commit > explicit branch/tag > pinned default."""
    return _env("GGUF_TRAIN_LLAMA_COMMIT") or _env("GGUF_TRAIN_LLAMA_BRANCH") or LLAMA_PINNED_COMMIT


def _log(msg: str) -> None:
    sys.stderr.write(f"[gguf_train] {msg}\n")
    sys.stderr.flush()


GT_SOURCES = [
    "csrc/gt_core.cpp",
    "csrc/gt_ops_math.cpp",
    "csrc/gt_ops_nn.cpp",
    "csrc/gt_ops_extra.cpp",
]


def lib_ext() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        return ".dylib"
    if sys_name == "Windows":
        return ".dll"
    return ".so"


def lib_prefix() -> str:
    return "" if platform.system() == "Windows" else "lib"


def lib_filename(stem: str) -> str:
    return f"{lib_prefix()}{stem}{lib_ext()}"


def _find_compilers():
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cc or not cxx:
        raise RuntimeError(
            "A C/C++ compiler (gcc/g++ or clang/clang++) is required to build gguf_train native libraries."
        )
    return cc, cxx


def _supports_openmp(cxx: str) -> bool:
    if platform.system() == "Darwin":
        return False
    try:
        res = subprocess.run(
            [cxx, "-fopenmp", "-x", "c++", "-", "-shared", "-o", os.devnull],
            input="int f(){return 0;}\n",
            text=True,
            capture_output=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False


def _supports_flag(compiler: str, flag: str) -> bool:
    try:
        res = subprocess.run(
            [compiler, flag, "-x", "c", "-", "-c", "-o", os.devnull],
            input="int f(void){return 0;}\n",
            text=True,
            capture_output=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False


def _download(url: str, dest: Path, timeout: int = 90) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_source_stamp(llama_dir: Path | None = None) -> dict | None:
    """Revision record of a tree fetched by this builder; None for a tree it does not manage."""
    return _read_json((llama_dir or LLAMA_DIR) / SOURCE_STAMP_NAME)


def _has_llama_tree(llama_dir: Path | None = None) -> bool:
    d = llama_dir or LLAMA_DIR
    return (d / "CMakeLists.txt").is_file() and (d / "ggml" / "src" / "ggml.c").is_file()


def _is_hex(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]+", s))


def _stamp_matches(stamp: dict, repo: str, ref: str) -> bool:
    if stamp.get("repo") != repo:
        return False
    commit = str(stamp.get("commit") or "")
    if ref == stamp.get("ref") or (commit and ref == commit):
        return True
    # An abbreviated sha naming the commit the tree already holds.
    return len(ref) >= 7 and _is_hex(ref) and commit.lower().startswith(ref.lower())


def _extract_archive(tar: tarfile.TarFile, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    # Interpreters without extraction filters: refuse anything that would land
    # outside dest and drop links/special files (the build needs neither).
    root = dest.resolve()
    members = []
    for m in tar.getmembers():
        target = (root / m.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"refusing to extract {m.name!r}: path escapes the archive root")
        if m.isfile() or m.isdir():
            members.append(m)
    tar.extractall(dest, members=members)


def _fetch_llama_archive(repo: str, ref: str) -> tuple[Path, str]:
    """Download and unpack <repo>@<ref>. Returns (unpacked tree, commit sha or "")."""
    archive_url = f"{repo}/archive/{ref}.tar.gz"
    tmp_tar = BUILD_DIR / ("llama-" + re.sub(r"[^A-Za-z0-9._-]", "_", ref) + ".tar.gz")
    extract_root = BUILD_DIR / "llama-download"
    attempts = 4
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            _download(archive_url, tmp_tar)
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            with tarfile.open(tmp_tar) as tar:
                # GitHub records the archived commit in the pax global header.
                commit = str(tar.pax_headers.get("comment", "")).strip()
                _extract_archive(tar, extract_root)
            tops = [d for d in extract_root.iterdir() if d.is_dir()]
            if len(tops) != 1 or not _has_llama_tree(tops[0]):
                raise RuntimeError("archive does not look like a llama.cpp source tree")
            return tops[0], (commit if _is_hex(commit) else "")
        except Exception as exc:  # noqa: BLE001 - retry then report
            last_err = exc
            code = getattr(exc, "code", None)
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                break  # e.g. 404 for an unknown revision: retrying cannot help
            if attempt + 1 < attempts:
                time.sleep(min(30, 2 ** attempt))
        finally:
            if tmp_tar.exists():
                tmp_tar.unlink()
    shutil.rmtree(extract_root, ignore_errors=True)
    raise RuntimeError(f"Failed to download llama.cpp revision {ref!r} from {archive_url}: {last_err}")


def ensure_llama_source(force_refresh: bool = False) -> Path:
    """Ensure third_party/llama.cpp holds the requested llama.cpp revision.

    The requested revision is GGUF_TRAIN_LLAMA_COMMIT, else GGUF_TRAIN_LLAMA_BRANCH,
    else the pinned LLAMA_PINNED_COMMIT. A tree fetched by this builder carries a
    stamp naming its revision and is re-fetched when a different one is requested.
    A tree without a stamp (placed there by hand, or by an older gguf_train) is
    used as-is unless a revision is requested explicitly or a refresh is forced,
    so vendored/offline builds keep working.
    """
    repo, ref = llama_repo(), llama_ref()
    explicit = bool(_env("GGUF_TRAIN_LLAMA_COMMIT") or _env("GGUF_TRAIN_LLAMA_BRANCH"))
    refresh = force_refresh or _env("GGUF_TRAIN_LLAMA_REFRESH", "0") == "1"
    try:
        where = LLAMA_DIR.relative_to(ROOT).as_posix()
    except ValueError:
        where = str(LLAMA_DIR)

    if _has_llama_tree() and not refresh:
        stamp = read_source_stamp()
        if stamp is None:
            if not explicit:
                _log(
                    f"using the existing {where} as-is; its revision is unknown, so this build is not "
                    f"pinned. Pass --refresh-llama to replace it with llama.cpp {LLAMA_PINNED_TAG}."
                )
                return LLAMA_DIR
            _log(f"{where} is of unknown revision but {ref} was requested; fetching it")
        elif _stamp_matches(stamp, repo, ref):
            return LLAMA_DIR
        else:
            _log(f"{where} holds {stamp.get('commit') or stamp.get('ref')} but {ref} is requested; fetching it")

    if (LLAMA_DIR / ".git").exists():
        raise RuntimeError(
            f"{LLAMA_DIR} is a git checkout, so it will not be overwritten. Check out the "
            f"revision you want there with git, or remove the directory to let the builder fetch {ref}."
        )

    label = f"{LLAMA_PINNED_TAG} ({ref[:12]})" if ref == LLAMA_PINNED_COMMIT else ref
    _log(f"fetching llama.cpp {label} from {repo}")
    # Download and unpack first so a failed fetch leaves any existing tree intact.
    top, commit = _fetch_llama_archive(repo, ref)
    if len(ref) == 40 and _is_hex(ref) and commit and commit.lower() != ref.lower():
        shutil.rmtree(top.parent, ignore_errors=True)
        raise RuntimeError(f"Requested llama.cpp commit {ref} but the archive contains {commit}")

    if LLAMA_DIR.exists():
        shutil.rmtree(LLAMA_DIR)
    LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(top), str(LLAMA_DIR))
    shutil.rmtree(top.parent, ignore_errors=True)
    (LLAMA_DIR / SOURCE_STAMP_NAME).write_text(
        json.dumps({"repo": repo, "ref": ref, "commit": commit}, indent=2, sort_keys=True) + "\n"
    )
    return LLAMA_DIR


def _needs_rebuild(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return True
    t_mtime = target.stat().st_mtime
    for s in sources:
        if s.exists() and s.stat().st_mtime > t_mtime:
            return True
    return False


def _cmake_version(cmake_file: Path, prefix: str) -> str | None:
    """Read set(<prefix>_VERSION_{MAJOR,MINOR,PATCH} n) from an upstream CMakeLists.txt."""
    try:
        text = cmake_file.read_text(errors="replace")
    except OSError:
        return None
    parts = []
    for comp in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"set\s*\(\s*{prefix}_VERSION_{comp}\s+(\d+)\s*\)", text)
        if not m:
            return None
        parts.append(m.group(1))
    return ".".join(parts)


def _write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def _ensure_generated_headers(gen_inc: Path, llama_dir: Path | None = None) -> None:
    """Generate ggml-version.h / llama-version.h describing the tree actually being built.

    Upstream produces these with CMake's configure_file(); mirror that by filling
    in its own templates with the versions declared in its CMakeLists.txt.
    """
    llama_dir = llama_dir or LLAMA_DIR
    gen_inc.mkdir(parents=True, exist_ok=True)
    stamp = read_source_stamp(llama_dir) or {}
    commit = str(stamp.get("commit") or stamp.get("ref") or "unknown")
    values = {
        "GGML_VERSION": _cmake_version(llama_dir / "ggml" / "CMakeLists.txt", "GGML") or "unknown",
        "GGML_BUILD_COMMIT": commit,
        "LLAMA_VERSION": _cmake_version(llama_dir / "CMakeLists.txt", "LLAMA") or "unknown",
        "LLAMA_BUILD_COMMIT": commit,
    }
    headers = (
        ("ggml-version.h", llama_dir / "ggml" / "src" / "ggml-version.h.in", ("GGML_VERSION", "GGML_COMMIT")),
        ("llama-version.h", llama_dir / "src" / "llama-version.h.in", ("LLAMA_VERSION", "LLAMA_COMMIT")),
    )
    for name, template, (ver_macro, commit_macro) in headers:
        if template.is_file():
            content = re.sub(r"@(\w+)@", lambda m: values.get(m.group(1), "unknown"), template.read_text())
        else:
            content = (
                "#pragma once\n"
                f'#define {ver_macro} "{values[ver_macro]}"\n'
                f'#define {commit_macro} "{commit}"\n'
            )
        _write_if_changed(gen_inc / name, content)


def _glob_sources(directory: Path, base: Path) -> list[str]:
    if not directory.is_dir():
        return []
    found = [f for pattern in ("*.c", "*.cpp") for f in directory.glob(pattern) if f.is_file()]
    return sorted(f.relative_to(base).as_posix() for f in found)


def _arch_dir(machine: str | None = None) -> str | None:
    mach = (machine or platform.machine()).lower()
    for names, arch in _ARCH_DIRS:
        if mach in names:
            return arch
    return None


def collect_ggml_sources(llama_dir: Path | None = None, machine: str | None = None) -> list[str]:
    """ggml core + CPU backend translation units present in the tree, relative to llama_dir."""
    llama_dir = llama_dir or LLAMA_DIR
    ggml_src = llama_dir / "ggml" / "src"
    cpu = ggml_src / "ggml-cpu"
    sources = _glob_sources(ggml_src, llama_dir)
    sources += _glob_sources(cpu, llama_dir)
    for sub in GGML_CPU_SUBDIRS:
        sources += _glob_sources(cpu / sub, llama_dir)
    arch = _arch_dir(machine)
    if arch:
        arch_dir = cpu / "arch" / arch
        sources += [(arch_dir / n).relative_to(llama_dir).as_posix() for n in _ARCH_FILES if (arch_dir / n).is_file()]
    return sources


def collect_llama_sources(llama_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """(core, per-architecture model) translation units of libllama, relative to llama_dir."""
    llama_dir = llama_dir or LLAMA_DIR
    core = _glob_sources(llama_dir / "src", llama_dir)
    models = _glob_sources(llama_dir / "src" / "models", llama_dir)
    return core, models


def _sync_build_stamp() -> bool:
    """Discard build products of another llama.cpp revision. Returns True if any were discarded.

    Freshness is otherwise judged by mtime, and files unpacked from an archive
    keep their commit timestamps, so a tree swapped for another revision can
    look *older* than the objects and libraries built from the previous one.
    The stamp is rewritten before compiling, which lets an interrupted build of
    the new revision resume instead of starting over.
    """
    current = read_source_stamp()
    built_from = _read_json(BUILD_STAMP) if BUILD_STAMP.exists() else None
    if current == built_from:
        # Same revision, or an unmanaged tree built before (mtimes are all there is to go on).
        return False
    for d in (BUILD_DIR / "obj", BUILD_DIR / "gen_src", BUILD_DIR / "include"):
        shutil.rmtree(d, ignore_errors=True)
    discarded = False
    for stem in ("ggml_train_ggml", "gguf_train", "llama_train_llama", "llama_bridge"):
        lib = OUT_DIR / lib_filename(stem)
        if lib.exists():
            lib.unlink()
            discarded = True
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if current is None:
        BUILD_STAMP.unlink()
    else:
        BUILD_STAMP.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return discarded


def _compile_unit(cmd: list[str], src: Path, obj: Path, force: bool) -> Path:
    """Compile src to obj with `cmd` (compiler + flags, without the -c/-o arguments)."""
    obj.parent.mkdir(parents=True, exist_ok=True)
    if not force and not _needs_rebuild(obj, [src]):
        return obj
    # Compile under a temporary name and rename on success: an interrupted compiler must
    # not leave a truncated object behind that the next (resumed) build takes for current.
    tmp = obj.with_name(obj.name + ".part")
    cmd = cmd + ["-c", str(src), "-o", str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Compilation failed for {src}:\nCMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
        )
    os.replace(tmp, obj)
    return obj


def _generate_llama_model_unity_chunks(gen_src: Path, batch_size: int = 16) -> list[Path]:
    """Group src/models/*.cpp into unity translation units (upstream uses the same batch size)."""
    _, models = collect_llama_sources()
    model_files = [LLAMA_DIR / m for m in models]
    gen_src.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    for idx in range(0, len(model_files), batch_size):
        group = model_files[idx : idx + batch_size]
        chunk_file = gen_src / f"llama_models_unity_{idx // batch_size:02d}.cpp"
        content = "".join(f'#include "{p.resolve().as_posix()}"\n' for p in group)
        _write_if_changed(chunk_file, content)
        chunks.append(chunk_file)
    # A previous build may have left more chunks behind than this tree needs.
    for leftover in gen_src.glob("llama_models_unity_*.cpp"):
        if leftover not in chunks:
            leftover.unlink()
    return chunks


def _rpath_flags() -> list[str]:
    sys_name = platform.system()
    if sys_name == "Darwin":
        return ["-Wl,-rpath,@loader_path"]
    if sys_name == "Linux":
        return ["-Wl,-rpath,$ORIGIN"]
    return []


def build_with_cmake(args) -> None:
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake was requested but is not installed.")
    ensure_llama_source(force_refresh=bool(getattr(args, "refresh_llama", False)))
    cm_build = ROOT / "build" / "cmake"
    cm_build.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_cmd = [
        cmake,
        "-S", str(CSRC_DIR),
        "-B", str(cm_build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_CUDA={'ON' if args.cuda else 'OFF'}",
        f"-DGGML_VULKAN={'ON' if args.vulkan else 'OFF'}",
        f"-DGGML_METAL={'ON' if args.metal else 'OFF'}",
    ]
    subprocess.run(cfg_cmd, check=True)
    jobs = str(args.jobs or os.cpu_count() or 2)
    subprocess.run([cmake, "--build", str(cm_build), "-j", jobs], check=True)

    ext = lib_ext()
    for p in cm_build.rglob(f"*{ext}*"):
        if p.is_file() or p.is_symlink():
            if any(k in p.name for k in ("ggml", "llama", "gguf_train")):
                dst = OUT_DIR / p.name
                shutil.copy2(p, dst)


def build_direct(
    jobs: int = 0,
    build_llama: bool = True,
    force: bool = False,
    refresh_llama: bool = False,
) -> dict[str, Path]:
    cc, cxx = _find_compilers()
    if jobs <= 0:
        jobs = max(1, int(os.environ.get("GGUF_TRAIN_BUILD_JOBS", "0")) or (os.cpu_count() or 2))

    ensure_llama_source(force_refresh=refresh_llama)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _sync_build_stamp():
        _log("llama.cpp revision changed since the last build; rebuilding the native libraries")
    gen_inc = BUILD_DIR / "include"
    gen_src = BUILD_DIR / "gen_src"
    obj_dir = BUILD_DIR / "obj"
    _ensure_generated_headers(gen_inc)

    ggml_out = OUT_DIR / lib_filename("ggml_train_ggml")
    gt_out = OUT_DIR / lib_filename("gguf_train")
    llama_out = OUT_DIR / lib_filename("llama_train_llama")
    bridge_out = OUT_DIR / lib_filename("llama_bridge")

    csrc_headers = list(CSRC_DIR.glob("*.h"))
    gt_srcs = [ROOT / s for s in GT_SOURCES]

    if not force:
        needed = (
            not ggml_out.exists()
            or _needs_rebuild(gt_out, gt_srcs + csrc_headers + [ggml_out])
        )
        if build_llama:
            needed = needed or not llama_out.exists() or _needs_rebuild(
                bridge_out,
                [CSRC_DIR / "llama_bridge.cpp", CSRC_DIR / "llama_bridge.h", CSRC_DIR / "gguf_train.h", gt_out, llama_out],
            )
        if not needed:
            return {
                "ggml": ggml_out,
                "gguf_train": gt_out,
                **({"llama": llama_out, "llama_bridge": bridge_out} if build_llama else {}),
            }

    has_omp = _supports_openmp(cxx)
    omp_flag = ["-fopenmp"] if has_omp else []
    omp_def = ["-DGGML_USE_OPENMP"] if has_omp else []
    arch_opt = ["-march=native"] if _supports_flag(cc, "-march=native") and platform.system() != "Darwin" else []

    common_posix_defs = [
        "-O3",
        "-fPIC",
        "-DNDEBUG",
        "-D_XOPEN_SOURCE=600",
        "-D_GNU_SOURCE",
    ] + arch_opt

    ggml_defs = common_posix_defs + omp_def + omp_flag + [
        "-DGGML_SHARED",
        "-DGGML_BUILD",
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        f"-I{gen_inc}",
        f"-I{GGML_DIR / 'include'}",
        f"-I{GGML_DIR / 'src'}",
        f"-I{GGML_DIR / 'src' / 'ggml-cpu'}",
    ]

    ggml_rel_sources = collect_ggml_sources()
    llama_core_sources, _ = collect_llama_sources()
    if not ggml_rel_sources or (build_llama and not llama_core_sources):
        raise RuntimeError(f"No llama.cpp/ggml sources found under {LLAMA_DIR}; the source tree looks incomplete.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # 1. Compile ggml objects
        ggml_objs: list[Path] = []
        ggml_futures = []
        if force or not ggml_out.exists() or _needs_rebuild(ggml_out, [LLAMA_DIR / s for s in ggml_rel_sources]):
            for rel in ggml_rel_sources:
                src = LLAMA_DIR / rel
                obj = obj_dir / (rel + ".o")
                ggml_objs.append(obj)
                is_c = src.suffix == ".c"
                comp = cc if is_c else cxx
                std = "-std=c11" if is_c else "-std=c++17"
                cmd = [comp, std] + ggml_defs
                ggml_futures.append(pool.submit(_compile_unit, cmd, src, obj, force))
            for f in ggml_futures:
                f.result()

            sys_libs = ["-ldl", "-lpthread", "-lm"] if platform.system() != "Windows" else []
            link_cmd = [cxx, "-shared"] + omp_flag + _rpath_flags() + ["-o", str(ggml_out)] + [str(o) for o in ggml_objs] + sys_libs
            res = subprocess.run(link_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Linking {ggml_out.name} failed:\n{res.stderr}\n{res.stdout}")

        # 2. Compile gguf_train objects
        gt_defs = common_posix_defs + omp_flag + [
            "-std=c++17",
            "-DGT_BUILD_SHARED",
            "-DGGML_SHARED",
            f"-I{CSRC_DIR}",
            f"-I{GGML_DIR / 'include'}",
        ]
        gt_objs: list[Path] = []
        gt_futures = []
        for s in gt_srcs:
            obj = obj_dir / "csrc" / (s.name + ".o")
            gt_objs.append(obj)
            cmd = [cxx] + gt_defs
            gt_futures.append(pool.submit(_compile_unit, cmd, s, obj, force or _needs_rebuild(obj, csrc_headers + [s])))
        for f in gt_futures:
            f.result()

        sys_libs = ["-lpthread", "-lm"] if platform.system() != "Windows" else []
        gt_link_cmd = (
            [cxx, "-shared"]
            + omp_flag
            + _rpath_flags()
            + ["-o", str(gt_out)]
            + [str(o) for o in gt_objs]
            + [f"-L{OUT_DIR}", "-lggml_train_ggml"]
            + sys_libs
        )
        res = subprocess.run(gt_link_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Linking {gt_out.name} failed:\n{res.stderr}\n{res.stdout}")

        # 3. Compile llama + llama_bridge if requested
        if build_llama:
            llama_defs = common_posix_defs + [
                "-std=c++17",
                "-DLLAMA_SHARED",
                "-DLLAMA_BUILD",
                "-DGGML_SHARED",
                f"-I{gen_inc}",
                f"-I{LLAMA_DIR / 'include'}",
                f"-I{LLAMA_DIR / 'src'}",
                f"-I{GGML_DIR / 'include'}",
            ]
            unity_chunks = _generate_llama_model_unity_chunks(gen_src, batch_size=16)
            llama_srcs = [LLAMA_DIR / s for s in llama_core_sources] + unity_chunks
            if force or not llama_out.exists() or _needs_rebuild(llama_out, llama_srcs):
                llama_objs: list[Path] = []
                llama_futures = []
                for s in llama_srcs:
                    rel_name = s.name if s.parent == gen_src else f"src_{s.name}"
                    obj = obj_dir / "llama" / (rel_name + ".o")
                    llama_objs.append(obj)
                    cmd = [cxx] + llama_defs
                    llama_futures.append(pool.submit(_compile_unit, cmd, s, obj, force))
                for f in llama_futures:
                    f.result()
                l_link_cmd = (
                    [cxx, "-shared"]
                    + _rpath_flags()
                    + ["-o", str(llama_out)]
                    + [str(o) for o in llama_objs]
                    + [f"-L{OUT_DIR}", "-lggml_train_ggml", "-lpthread", "-lm"]
                )
                res = subprocess.run(l_link_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"Linking {llama_out.name} failed:\n{res.stderr}\n{res.stdout}")

            bridge_src = CSRC_DIR / "llama_bridge.cpp"
            bridge_obj = obj_dir / "csrc" / "llama_bridge.cpp.o"
            bridge_defs = common_posix_defs + [
                "-std=c++17",
                "-DGT_BUILD_SHARED",
                "-DGGML_SHARED",
                "-DLLAMA_SHARED",
                f"-I{CSRC_DIR}",
                f"-I{LLAMA_DIR / 'include'}",
                f"-I{GGML_DIR / 'include'}",
            ]
            b_cmd = [cxx] + bridge_defs
            _compile_unit(b_cmd, bridge_src, bridge_obj, force or _needs_rebuild(bridge_obj, [bridge_src] + csrc_headers))
            b_link_cmd = (
                [cxx, "-shared"]
                + _rpath_flags()
                + ["-o", str(bridge_out), str(bridge_obj)]
                + [f"-L{OUT_DIR}", "-lgguf_train", "-lllama_train_llama", "-lggml_train_ggml", "-lpthread", "-lm"]
            )
            res = subprocess.run(b_link_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Linking {bridge_out.name} failed:\n{res.stderr}\n{res.stdout}")

    result = {"ggml": ggml_out, "gguf_train": gt_out}
    if build_llama:
        result["llama"] = llama_out
        result["llama_bridge"] = bridge_out
    return result


def main():
    parser = argparse.ArgumentParser(description="Build gguf_train native libraries on top of a pinned llama.cpp")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="Parallel compile jobs")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--no-llama", action="store_true", help="Skip building llama_bridge (core ggml tensor engine only)")
    parser.add_argument(
        "--refresh-llama",
        action="store_true",
        help="Re-download the configured llama.cpp revision even if a local copy exists",
    )
    parser.add_argument("--cmake", action="store_true", help="Use CMake instead of direct compiler invocation")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA via CMake")
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan via CMake")
    parser.add_argument("--metal", action="store_true", help="Enable Metal via CMake")
    args = parser.parse_args()

    if args.cmake or args.cuda or args.vulkan or args.metal:
        build_with_cmake(args)
        print(f"Built native libraries with CMake in {OUT_DIR}")
    else:
        libs = build_direct(
            jobs=args.jobs,
            build_llama=not args.no_llama,
            force=args.force,
            refresh_llama=args.refresh_llama,
        )
        stamp = read_source_stamp()
        if stamp:
            print(f"llama.cpp revision: {stamp.get('commit') or stamp.get('ref')}")
        print("Built native libraries:")
        for k, v in libs.items():
            print(f"  {k}: {v} ({v.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
