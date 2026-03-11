from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


logger = logging.getLogger(__name__)


def _default_config_path(project_root: Path | None = None) -> Path:
    """
    Return the default path to properties.yaml under the given project root.
    """
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "config" / "properties.yaml"


def load_properties(config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load configuration properties from a YAML file.

    - If config_path is not provided, looks for config/properties.yaml
      relative to the project root.
    - Returns an empty dict if the file does not exist or is empty.
    """
    path = config_path or _default_config_path()
    if not path.exists():
        logger.info("Config file %s not found; using empty properties.", path)
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            logger.warning(
                "Config file %s did not contain a top-level mapping; ignoring.", path
            )
            return {}
        return data
    except Exception as exc:
        logger.exception("Failed to load configuration from %s: %s", path, exc)
        return {}

