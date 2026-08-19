"""Run every M1 test module. No GPU, camera, or weights required."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_state_machine.py",
    "test_associator.py",
    "test_pipeline_smoke.py",
    "test_backend_m2.py",
    "test_agent_m3.py",
    "test_notify.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    failed = []
    for name in TESTS:
        print(f"\n=== {name} " + "=" * (60 - len(name)), flush=True)
        result = subprocess.run([sys.executable, str(here / name)], stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            failed.append(name)

    print("\n" + "=" * 68, flush=True)
    print(f"FAILED: {', '.join(failed)}" if failed else "all modules passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
