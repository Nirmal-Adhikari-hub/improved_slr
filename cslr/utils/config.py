import json
import yaml
from pathlib import Path
from typing import Any, Dict


def _dot_set(d: Dict[str, Any], key: str, value: Any):
    """
    Internal use
    Args:
        d (dict): The dictionary to update.
        key (str): Dot-separated key path (eg 'a.b.c')
        value: The vlaue to set at the given path
    """
    parts = key.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def update_config(cfg: Dict[str, Any], overrides: Dict[str,Any]) -> Dict[str,Any]:
    """
    Args:
        cfg (dict): The original configuration dictionary.
        overrides (dict): A dictionary of overrides, where keys are dot-separated paths.

    Returns:
        dict: The updated configuration dictionary with overrides applied.
    """
    out = dict(cfg)
    for k, v in overrides.items():
        _dot_set(out, k, v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed, but a YAML config was given.")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config file type: {p.suffix}")