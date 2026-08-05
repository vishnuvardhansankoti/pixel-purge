"""Configuration management — reads ``~/.pixel-purge/config.toml`` with sane defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(os.path.expanduser("~/.pixel-purge"))
DEFAULT_CONFIG_PATH = APP_DIR / "config.toml"
DEFAULT_DB_PATH = APP_DIR / "manifest.db"


@dataclass
class DedupConfig:
    gps_radius_meters: float = 100.0
    time_window_minutes: int = 30
    # Tightened from the PRD's 10 to reduce false-merge risk on text-dense images [H1].
    hamming_threshold: int = 8
    # Near-exact threshold applied when either candidate is text-heavy
    # (screenshots/documents), so distinct text images are not merged [H1].
    text_hamming_threshold: int = 2


@dataclass
class DeltaConfig:
    trip_distance_miles: float = 50.0
    model: str = "ViT-B-32"
    device: str = "auto"
    notify: bool = True


@dataclass
class Config:
    db_path: Path = DEFAULT_DB_PATH
    log_level: str = "INFO"
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    dedup: DedupConfig = field(default_factory=DedupConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        cfg = cls()
        if not path.exists():
            return cfg

        with open(path, "rb") as f:
            data = tomllib.load(f)

        general = data.get("general", {})
        if "db_path" in general:
            cfg.db_path = Path(os.path.expanduser(general["db_path"]))
        cfg.log_level = general.get("log_level", cfg.log_level)

        home = data.get("home_base", {})
        cfg.home_latitude = float(home.get("latitude", cfg.home_latitude))
        cfg.home_longitude = float(home.get("longitude", cfg.home_longitude))

        d = data.get("dedup", {})
        cfg.dedup = DedupConfig(
            gps_radius_meters=float(d.get("gps_radius_meters", DedupConfig.gps_radius_meters)),
            time_window_minutes=int(d.get("time_window_minutes", DedupConfig.time_window_minutes)),
            hamming_threshold=int(d.get("hamming_threshold", DedupConfig.hamming_threshold)),
            text_hamming_threshold=int(
                d.get("text_hamming_threshold", DedupConfig.text_hamming_threshold)
            ),
        )

        dl = data.get("delta", {})
        cfg.delta = DeltaConfig(
            trip_distance_miles=float(dl.get("trip_distance_miles", DeltaConfig.trip_distance_miles)),
            model=dl.get("model", DeltaConfig.model),
            device=dl.get("device", DeltaConfig.device),
            notify=bool(dl.get("notify", DeltaConfig.notify)),
        )
        return cfg
