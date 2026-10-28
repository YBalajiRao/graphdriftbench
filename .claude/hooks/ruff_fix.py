"""PostToolUse hook: run ruff fix + format, never fail (Windows-safe, no shell syntax)."""

import shutil
import subprocess
import sys


def main() -> int:
    ruff = shutil.which("ruff")
    cmd = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    for args in (["check", "--fix", "--quiet", "."], ["format", "--quiet", "."]):
        try:
            subprocess.run(cmd + args, check=False, capture_output=True)
        except OSError:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
