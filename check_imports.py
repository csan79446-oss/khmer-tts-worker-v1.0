"""
Build-time import verification for the worker image (also handy for debugging).

  python /app/check_imports.py            # report; exit 1 only if CORE fails
  python /app/check_imports.py --strict   # exit 1 if anything fails

CORE     - without these the image cannot work at all, so a failure fails the
           build.
OPTIONAL - reported with a full traceback but do NOT fail the build. The worker
           detects a missing engine at runtime and reports it loudly (labelled
           Edge-TTS fallback, `engine` field in every response). A completed
           build tells us exactly which import broke; an aborted build tells us
           nothing at all.
"""
import importlib
import sys
import traceback

CORE = ("torch", "torchaudio", "numpy", "soundfile", "edge_tts", "runpod")
OPTIONAL = ("librosa", "voxcpm")


def check(name: str) -> bool:
    """Import `name`, print the outcome and, on failure, the full traceback."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - we want to report *everything*
        print(f"[FAIL] {name:<12} {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print(f"[ OK ] {name:<12} {getattr(module, '__version__', 'unknown')}")
    return True


def main() -> int:
    strict = "--strict" in sys.argv
    print("=" * 72)
    print("worker image import check")
    print("=" * 72)

    core_results = [check(name) for name in CORE]  # check every one, no short-circuit
    core_ok = all(core_results)

    print("-" * 72)
    optional_results = [check(name) for name in OPTIONAL]
    optional_ok = all(optional_results)

    print("=" * 72)
    if not core_ok:
        print("CORE IMPORTS FAILED - this image is not usable.")
        return 1
    if not optional_ok and strict:
        print("=" * 72)
        print("OPTIONAL IMPORT FAILED (strict mode) - build aborted.")
        print("This usually means a dependency pin conflicts with voxcpm 2.0.3.")
        print("Check the traceback for [FAIL] voxcpm / librosa above.")
        return 1
    if not optional_ok:
        print("WARNING: librosa and/or voxcpm did not import.")
        print("         Real VoxCPM synthesis is unavailable; the worker will serve")
        print("         its labelled Edge-TTS fallback. Paste the traceback above to")
        print("         get the dependency pin fixed.")
        return 1 if strict else 0
    print("All imports OK - real VoxCPM synthesis is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
