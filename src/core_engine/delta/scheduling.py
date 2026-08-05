"""Generate a launchd agent to run the monthly delta locally (macOS).

Writes ~/Library/LaunchAgents/com.pixelpurge.delta.plist. The plist runs
`pixel-purge delta <source>` on a monthly schedule. Installation (load) is left
to the user via `launchctl` so nothing is registered without consent.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

LABEL = "com.pixelpurge.delta"
LAUNCH_AGENTS_DIR = Path(os.path.expanduser("~/Library/LaunchAgents"))


def build_plist(
    source: Path,
    executable: str,
    day: int = 1,
    hour: int = 9,
    notify: bool = True,
) -> dict:
    """Build the launchd property-list dict for the monthly delta run."""
    args = [executable, "delta", str(source)]
    if notify:
        args.append("--notify")
    return {
        "Label": LABEL,
        "ProgramArguments": args,
        "StartCalendarInterval": {"Day": int(day), "Hour": int(hour), "Minute": 0},
        "RunAtLoad": False,
        "StandardOutPath": str(Path.home() / ".pixel-purge" / "delta.log"),
        "StandardErrorPath": str(Path.home() / ".pixel-purge" / "delta.err.log"),
    }


def install_agent(
    source: Path,
    executable: str | None = None,
    day: int = 1,
    hour: int = 9,
    notify: bool = True,
    agents_dir: Path | None = None,
) -> Path:
    """Write the plist and return its path (does not `launchctl load`)."""
    import shutil
    import sys

    executable = executable or shutil.which("pixel-purge") or f"{sys.prefix}/bin/pixel-purge"
    agents_dir = agents_dir or LAUNCH_AGENTS_DIR
    agents_dir.mkdir(parents=True, exist_ok=True)
    plist_path = agents_dir / f"{LABEL}.plist"
    with open(plist_path, "wb") as f:
        plistlib.dump(build_plist(Path(source), executable, day, hour, notify), f)
    return plist_path


def load_hint(plist_path: Path) -> str:
    return (
        f"Wrote {plist_path}\n"
        f"Enable it with:  launchctl load {plist_path}\n"
        f"Disable it with: launchctl unload {plist_path}"
    )
