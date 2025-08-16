from __future__ import annotations
from cslr.utils.config import load_config, update_config

def _autotype(val: str):
    low = val.lower()
    if low in ("true","false"): return low == "true"
    try: return int(val)
    except: pass
    try: return float(val)
    except: pass
    try:
        import json; return json.loads(val)
    except: pass
    return val


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    if not pairs: return cfg
    if len(pairs) % 2 != 0:
        raise SystemExit("ERROR: --override expects pairs: <key> <value> ...")
    return update_config(
        cfg,
        {pairs[i]: _autotype(pairs[i+1]) for i in range(0, len(pairs), 2)}
    )