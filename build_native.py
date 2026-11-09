#!/usr/bin/env python3
"""
Dependency-free native builder for gguf_train.
Compiles vendored third_party/llama.cpp (ggml + llama) and csrc/ into
shared libraries under gguf_train/_libs/ using gcc/g++ or clang/clang++ directly
(no CMake or Python.h required), with optional CMake GPU backend support.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
GGML_DIR = LLAMA_DIR / "ggml"
OUT_DIR = ROOT / "gguf_train" / "_libs"
BUILD_DIR = ROOT / "build" / "native"

GGML_BASE_SOURCES = [
    "ggml/src/ggml.c",
    "ggml/src/ggml.cpp",
    "ggml/src/ggml-alloc.c",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-opt.cpp",
    "ggml/src/ggml-threading.cpp",
    "ggml/src/ggml-quants.c",
    "ggml/src/gguf.cpp",
    "ggml/src/ggml-backend-dl.cpp",
    "ggml/src/ggml-backend-reg.cpp",
]

GGML_CPU_SOURCES = [
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "ggml/src/ggml-cpu/repack.cpp",
    "ggml/src/ggml-cpu/hbm.cpp",
    "ggml/src/ggml-cpu/quants.c",
    "ggml/src/ggml-cpu/traits.cpp",
    "ggml/src/ggml-cpu/amx/amx.cpp",
    "ggml/src/ggml-cpu/amx/mmq.cpp",
    "ggml/src/ggml-cpu/tiled/tiled.cpp",
    "ggml/src/ggml-cpu/tiled/tiled-kernel.cpp",
    "ggml/src/ggml-cpu/binary-ops.cpp",
    "ggml/src/ggml-cpu/unary-ops.cpp",
    "ggml/src/ggml-cpu/vec.cpp",
    "ggml/src/ggml-cpu/ops.cpp",
]

LLAMA_CORE_SOURCES = [
    "src/llama.cpp",
    "src/llama-adapter.cpp",
    "src/llama-arch.cpp",
    "src/llama-batch.cpp",
    "src/llama-chat.cpp",
    "src/llama-context.cpp",
    "src/llama-cparams.cpp",
    "src/llama-grammar.cpp",
    "src/llama-graph.cpp",
    "src/llama-hparams.cpp",
    "src/llama-impl.cpp",
    "src/llama-io.cpp",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache-iswa.cpp",
    "src/llama-kv-cache-dsa.cpp",
    "src/llama-kv-cache-dsa-iswa.cpp",
    "src/llama-kv-cache-msa.cpp",
    "src/llama-kv-cache-dsv4.cpp",
    "src/llama-memory.cpp",
    "src/llama-moe-cache.cpp",
    "src/llama-memory-hybrid.cpp",
    "src/llama-memory-hybrid-iswa.cpp",
    "src/llama-memory-hybrid-idx.cpp",
    "src/llama-memory-recurrent.cpp",
    "src/llama-mmap.cpp",
    "src/llama-model-loader.cpp",
    "src/llama-model-saver.cpp",
    "src/llama-model.cpp",
    "src/llama-quant.cpp",
    "src/llama-sampler.cpp",
    "src/llama-vocab.cpp",
    "src/unicode-data.cpp",
    "src/unicode.cpp",
]

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


def _needs_rebuild(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return True
    t_mtime = target.stat().st_mtime
    for s in sources:
        if s.exists() and s.stat().st_mtime > t_mtime:
            return True
    return False


def _ensure_generated_headers(gen_inc: Path):
    gen_inc.mkdir(parents=True, exist_ok=True)
    ggml_ver = gen_inc / "ggml-version.h"
    if not ggml_ver.exists():
        ggml_ver.write_text(
            '#pragma once\n#define GGML_VERSION "0.29.0"\n#define GGML_COMMIT "vendored"\n'
        )
    llama_ver = gen_inc / "llama-version.h"
    if not llama_ver.exists():
        llama_ver.write_text(
            '#pragma once\n#define LLAMA_VERSION "0.29.0"\n#define LLAMA_COMMIT "vendored"\n'
        )


def _arch_sources() -> list[str]:
    mach = platform.machine().lower()
    if mach in ("x86_64", "amd64", "i686", "x86"):
        return [
            "ggml/src/ggml-cpu/arch/x86/quants.c",
            "ggml/src/ggml-cpu/arch/x86/repack.cpp",
        ]
    if mach in ("arm64", "aarch64"):
        return [
            "ggml/src/ggml-cpu/arch/arm/quants.c",
            "ggml/src/ggml-cpu/arch/arm/repack.cpp",
        ]
    if "ppc64" in mach:
        return [
            "ggml/src/ggml-cpu/arch/powerpc/quants.c",
            "ggml/src/ggml-cpu/arch/powerpc/repack.cpp",
        ]
    if "riscv" in mach:
        return [
            "ggml/src/ggml-cpu/arch/riscv/quants.c",
            "ggml/src/ggml-cpu/arch/riscv/repack.cpp",
        ]
    if "s390" in mach:
        return ["ggml/src/ggml-cpu/arch/s390/quants.c"]
    if "loongarch" in mach:
        return [
            "ggml/src/ggml-cpu/arch/loongarch/quants.c",
            "ggml/src/ggml-cpu/arch/loongarch/repack.cpp",
        ]
    return []


def _compile_unit(cmd: list[str], src: Path, obj: Path, force: bool) -> Path:
    obj.parent.mkdir(parents=True, exist_ok=True)
    if not force and not _needs_rebuild(obj, [src]):
        return obj
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Compilation failed for {src}:\nCMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
        )
    return obj


def _generate_llama_model_unity_chunks(gen_src: Path, batch_size: int = 16) -> list[Path]:
    models_dir = LLAMA_DIR / "src" / "models"
    model_files = sorted(models_dir.glob("*.cpp"))
    gen_src.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    for idx in range(0, len(model_files), batch_size):
        group = model_files[idx : idx + batch_size]
        chunk_file = gen_src / f"llama_models_unity_{idx // batch_size:02d}.cpp"
        content = "".join(f'#include "{p.resolve()}"\n' for p in group)
        if not chunk_file.exists() or chunk_file.read_text() != content:
            chunk_file.write_text(content)
        chunks.append(chunk_file)
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


def build_direct(jobs: int = 0, build_llama: bool = True, force: bool = False) -> dict[str, Path]:
    cc, cxx = _find_compilers()
    if jobs <= 0:
        jobs = max(1, int(os.environ.get("GGUF_TRAIN_BUILD_JOBS", "0")) or (os.cpu_count() or 2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
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

    ggml_rel_sources = GGML_BASE_SOURCES + GGML_CPU_SOURCES + _arch_sources()
    tasks = []

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
                cmd = [comp, std] + ggml_defs + ["-c", str(src), "-o", str(obj)]
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
            cmd = [cxx] + gt_defs + ["-c", str(s), "-o", str(obj)]
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
            llama_srcs = [LLAMA_DIR / s for s in LLAMA_CORE_SOURCES] + unity_chunks
            if force or not llama_out.exists() or _needs_rebuild(llama_out, llama_srcs):
                llama_objs: list[Path] = []
                llama_futures = []
                for s in llama_srcs:
                    rel_name = s.name if s.parent == gen_src else f"src_{s.name}"
                    obj = obj_dir / "llama" / (rel_name + ".o")
                    llama_objs.append(obj)
                    cmd = [cxx] + llama_defs + ["-c", str(s), "-o", str(obj)]
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
            b_cmd = [cxx] + bridge_defs + ["-c", str(bridge_src), "-o", str(bridge_obj)]
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
    parser = argparse.ArgumentParser(description="Build gguf_train native libraries on top of vendored llama.cpp")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="Parallel compile jobs")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--no-llama", action="store_true", help="Skip building llama_bridge (core ggml tensor engine only)")
    parser.add_argument("--cmake", action="store_true", help="Use CMake instead of direct compiler invocation")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA via CMake")
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan via CMake")
    parser.add_argument("--metal", action="store_true", help="Enable Metal via CMake")
    args = parser.parse_args()

    if args.cmake or args.cuda or args.vulkan or args.metal:
        build_with_cmake(args)
        print(f"Built native libraries with CMake in {OUT_DIR}")
    else:
        libs = build_direct(jobs=args.jobs, build_llama=not args.no_llama, force=args.force)
        print("Built native libraries:")
        for k, v in libs.items():
            print(f"  {k}: {v} ({v.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
