"""Check that the llama.cpp (GGUF) OCR backend is ready to run on this machine.

Checks, without downloading anything:

1. the configuration loads and selects the `llamacpp` recognizer,
2. llama-server is found and is new enough for Qwen3.5 models (b7990+),
3. both model files exist, are real GGUF files, are the right two files and
   are not swapped,
4. the machine's RAM (and, if `nvidia-smi` is present, its GPU) next to the
   model's size.

With --start it also launches the server exactly as the service would and
confirms it loaded the vision projector; with --image it reads one image.

    python tools/check_llamacpp_setup.py
    python tools/check_llamacpp_setup.py --start
    python tools/check_llamacpp_setup.py --image path\\to\\cell.png --kind number
    python tools/check_llamacpp_setup.py --config config/nuqta_config.yaml

Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from core.domain.roles import ContentKind  # noqa: E402
from ocr.llamacpp.gguf import check_vision_model_pair  # noqa: E402
from ocr.llamacpp.server import MIN_QWEN35_BUILD, parse_build_number, resolve_executable  # noqa: E402

OK, FAIL, WARN, INFO = "[ OK ]", "[FAIL]", "[WARN]", "[INFO]"


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, message: str) -> None:
        if status == FAIL:
            self.failed = True
        print(f"{status} {message}", flush=True)


def _gib(size: int) -> str:
    return f"{size / 1024**3:.2f} GiB"


def total_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - informational only
        return None
    return None


def gpu_summary() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def server_version(executable: str) -> str:
    try:
        out = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(could not run: {exc})"
    return (out.stdout + out.stderr).strip()


def check(config_path: Path, start: bool, image: Path | None, kind: str) -> int:
    report = Report()

    # 1. configuration -----------------------------------------------------
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        report.line(FAIL, f"Configuration {config_path} does not load: {exc}")
        return 1
    recognizer = config.ocr.recognizer
    if recognizer is None or recognizer.name != "llamacpp":
        report.line(FAIL, f"{config_path.name} does not select the llamacpp recognizer "
                          f"(ocr.recognizer is {recognizer.name if recognizer else 'not set'}).")
        return 1
    params = dict(recognizer.params)
    report.line(OK, f"Configuration {config_path.name} selects the llamacpp recognizer (flow: {config.flow.name}).")

    # 2. llama-server ------------------------------------------------------
    executable = None
    try:
        executable = resolve_executable(params.get("server_path", "llama-server"))
        version = server_version(executable)
        build = parse_build_number(version)
        report.line(OK, f"llama-server: {executable}")
        if build is None:
            report.line(WARN, f"Could not read the build number from: {version.splitlines()[0] if version else '(no output)'}")
        elif build < MIN_QWEN35_BUILD:
            report.line(FAIL, f"llama.cpp build {build} is too old: Qwen3.5 models need b{MIN_QWEN35_BUILD} or newer.")
        else:
            report.line(OK, f"llama.cpp build {build} (>= b{MIN_QWEN35_BUILD}, supports Qwen3.5).")
        gpu_lines = [line for line in version.splitlines() if "CUDA" in line or "Vulkan" in line or "device" in line.lower()]
        for line in gpu_lines[:4]:
            report.line(INFO, f"  {line.strip()}")
    except FileNotFoundError as exc:
        report.line(FAIL, str(exc))

    # 3. model files -------------------------------------------------------
    model_bytes = 0
    try:
        def project_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT / path

        model, projector = check_vision_model_pair(
            project_path(params["model_path"]), project_path(params["mmproj_path"])
        )
        model_bytes = model.size_bytes + projector.size_bytes
        report.line(OK, f"Model:     {model.path} ({_gib(model.size_bytes)}, architecture "
                        f"'{model.architecture}', {model.metadata.get('general.name', 'unnamed')})")
        report.line(OK, f"Projector: {projector.path} ({_gib(projector.size_bytes)}, "
                        f"{projector.projector_type or 'projector type not stated'})")
        if model.architecture != "qwen35":
            report.line(WARN, f"The model's architecture is '{model.architecture}', not 'qwen35'. That is fine "
                              "for another llama.cpp vision model, but it is not the Nuqta 4B build.")
    except KeyError as exc:
        report.line(FAIL, f"The recognizer configuration has no {exc}.")
    except Exception as exc:  # noqa: BLE001 - FileNotFoundError, ModelFileError
        report.line(FAIL, str(exc))

    # 4. hardware ----------------------------------------------------------
    ram = total_ram_bytes()
    if ram:
        status = WARN if model_bytes and model_bytes > ram * 0.6 else INFO
        report.line(status, f"RAM: {_gib(ram)} total; model files {_gib(model_bytes)}.")
    gpu = gpu_summary()
    if gpu:
        report.line(INFO, f"GPU (name, total, free): {gpu}. gpu_layers '{params.get('gpu_layers', 'auto')}' -- "
                          "'auto' fits what the free VRAM holds and keeps the rest on the CPU.")
    else:
        report.line(INFO, "No NVIDIA GPU found (or nvidia-smi is not on PATH); the model will run on the CPU "
                          "unless this llama.cpp build uses another GPU backend.")

    if report.failed or not (start or image):
        print("\nAll checks passed." if not report.failed else "\nSome checks failed; see above.")
        return 1 if report.failed else 0

    # 5. start the server (and read) ---------------------------------------
    from ocr.recognizers.registry import recognizer_registry
    import ocr.recognizers  # noqa: F401  (registers the recognizers)

    print("\nStarting llama-server as the service would (the first load can take a minute)...", flush=True)
    started = time.monotonic()
    try:
        built = recognizer_registry.create("llamacpp", **params)
    except Exception as exc:  # noqa: BLE001
        report.line(FAIL, f"The server did not start: {exc}")
        return 1
    try:
        report.line(OK, f"llama-server ready with image support in {time.monotonic() - started:.1f}s "
                        f"({built.server.build_info}).")
        # What the server said about where the model went. Older builds print
        # "offloaded N/M layers to GPU"; newer ones are quieter at the default
        # verbosity, and say at least which multimodal model loaded.
        placement = [
            line.strip() for line in built.server.log_tail(1000).splitlines()
            if any(key in line for key in (
                "offloaded", "using device", "ggml_cuda_init", "ggml_vulkan:", "CLIP using",
                "loaded multimodal", "n_ctx_slot", "n_threads",
            ))
        ]
        for line in placement[:8]:
            report.line(INFO, f"  server: {line}")
        if image is not None:
            from PIL import Image

            started = time.monotonic()
            reading = built._backend.read(Image.open(image).convert("RGB"), kind)
            confidence = "unknown" if reading.confidence is None else f"{reading.confidence:.3f}"
            report.line(OK, f"Read {image.name} as {kind} in {time.monotonic() - started:.1f}s: "
                            f"{reading.text!r} (confidence {confidence}).")
    except Exception as exc:  # noqa: BLE001
        report.line(FAIL, f"Reading failed: {exc}")
    finally:
        built.close()

    print("\nAll checks passed." if not report.failed else "\nSome checks failed; see above.")
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/nuqta_config.yaml")
    parser.add_argument("--start", action="store_true", help="also launch llama-server and confirm image support")
    parser.add_argument("--image", type=Path, help="also read this image (implies --start)")
    parser.add_argument("--kind", default="any", choices=[k.value for k in ContentKind],
                        help="content kind to read --image as (selects the prompt and constraint)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    return check(config_path, args.start, args.image, args.kind)


if __name__ == "__main__":
    sys.exit(main())
