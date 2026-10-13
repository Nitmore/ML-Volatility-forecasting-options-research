"""Config loading and repository paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` until the directory containing config.yaml is found."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    raise FileNotFoundError("config.yaml not found in this directory or any parent.")


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    figures: Path
    results: Path

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.figures, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(root: Path | None = None) -> tuple[dict, Paths]:
    root = root or find_root()
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    paths = Paths(
        root=root,
        raw=root / p["raw"],
        processed=root / p["processed"],
        figures=root / p["figures"],
        results=root / p["results"],
    ).ensure()
    return cfg, paths
