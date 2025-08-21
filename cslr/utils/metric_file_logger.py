from pathlib import Path
from typing import Dict
import csv


class ScalarFileLogger:
    """
    Lightweight CSV writer for split metrics (train/dev/test).
    Also auto-updates simple PNG plots if matplotlib is available.
    """
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.files = {
            "train": self.base / "train_metrics.csv",
            "dev": self.base / "dev_metrics.csv",
            "test": self.base / "test_metrics.csv",
        }
        self._mpl_ok = None

    def log(self, split: str, row: Dict[str, float]):
        if "epoch" not in row:
            raise ValueError(f"[{str(Path(__file__)).upper()}] row must include 'epoch'")
        path = self.files[split]
        exists = path.exists()
        with path.open("a", newline="") as f:
            keys = ["epoch"] + sorted(k for k in row.keys() if k != "epoch")
            w = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                w.writeheader()
            w.writerow(row)
        self._maybe_plot(split)
        
    def _maybe_plot(self, split: str):
        if self._mpl_ok is False:
            return
        try:
            import matplotlib.pyplot as plt
            xs, series = [], {}
            with self.files[split].open() as f:
                r = csv.DictReader(f)
                for rec in r:
                    xs.append(int(rec["epoch"]))
                    for k, v in rec.items():
                        if k == "epoch": continue
                        series.setdefault(k, []).append(float(v) if v != "" else float('nan'))
            for k, ys in series.items():
                plt.figure()
                plt.plot(xs, ys)
                plt.title(f"{split} - {k}")
                plt.xlabel("epoch"); plt.ylabel(k)
                plt.tight_layout()
                plt.savefig(self.base / f"{split}_{k}.png")
                plt.close()
            self._mpl_ok = True
        except Exception:
            self._mpl_ok =  False