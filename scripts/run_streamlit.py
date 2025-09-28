"""
Run the Streamlit dashboard for the churn project.

Usage:
  python scripts/run_streamlit.py [--port 8501] [--address 0.0.0.0]

This script auto-resolves the path to visualisation/streamlit.py relative to the project root
and uses either the 'streamlit' executable or 'python -m streamlit' as a fallback.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    # scripts/ -> project root is parent
    return Path(__file__).resolve().parent.parent


def get_streamlit_entrypoint(root: Path) -> Path:
    return root / "visualisation" / "streamlit.py"


def main():
    parser = argparse.ArgumentParser(description="Launch Streamlit app")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument(
        "--address", type=str, default="localhost", help="Server address (0.0.0.0 for LAN access)"
    )
    args, extra = parser.parse_known_args()

    root = find_project_root()
    entrypoint = get_streamlit_entrypoint(root)

    if not entrypoint.exists():
        print(f"Error: Streamlit entrypoint not found: {entrypoint}")
        sys.exit(1)

    # Prefer streamlit executable if available, otherwise fallback to python -m streamlit
    streamlit_cmd = shutil.which("streamlit")
    if streamlit_cmd is None:
        cmd = [sys.executable, "-m", "streamlit", "run", str(entrypoint), "--server.port", str(args.port), "--server.address", args.address]
    else:
        cmd = [streamlit_cmd, "run", str(entrypoint), "--server.port", str(args.port), "--server.address", args.address]

    # Allow passing through extra CLI args to Streamlit if needed
    if extra:
        cmd.extend(extra)

    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = os.environ.copy()
    # Recommended for wide layout consistency
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("Streamlit exited with error code:", e.returncode)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
