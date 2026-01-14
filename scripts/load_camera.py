from pathlib import Path
import pandas as pd

def load_camera(camera, day_dir):
    path = day_dir / f"{camera}.csv"
    if not path.exists():
        return None

    return pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )[["timestamp", "ocupada"]]
